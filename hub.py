"""
Neural Flight Hub: one place to train, watch progress, and make replays.

    python hub.py            # opens http://127.0.0.1:8765 in your browser
    python hub.py --port 9000 --no-browser

Uses only the Python standard library. Training and replays run as the same
`train.py` / `replay.py` you would run by hand, so everything still works from
the command line too.
"""
import argparse
import ast
import json
import queue
import os
import re

import signal
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent
WINDOWS = os.name == "nt"
RUNS_DIR = ROOT / "runs"
REPLAY_DIR = ROOT / "replays"
PY = sys.executable

LOG_RE = re.compile(
    r"it\s+(\d+)\s*\|\s*sps\s+([\d.]+)\s*\|\s*ep_ret\s+(-?[\d.]+)\s*\|\s*ep_len\s+([\d.]+)\s*\|\s*"
    r"rings/ep\s+([\d.]+)\s*\|\s*max\s+([\d.]+)\s*\|\s*ring_r\s+([\d.]+)\s*\|\s*std\s+([\d.]+)\s*\|\s*lr\s+([\d.e+-]+)"
    r"(?:\s*\|\s*power\s+([\d.]+))?"
)
NAME_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,63}$")   # no leading dot: ".." would escape runs/


# --------------------------------------------------------------------- runs & files
def load_species():
    """Read SPECIES_INFO from bird_env.py without importing torch (keeps the hub instant)."""
    tree = ast.parse((ROOT / "bird_env.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "SPECIES_INFO" for t in node.targets):
            return ast.literal_eval(node.value)
    return {"hawk": "The original bird"}


SPECIES = load_species()


def run_config(d):
    """Which bird a run trains. Runs from before species existed are the massless-wing hawk."""
    try:
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {}
    physics = cfg.get("physics") or ("wings-v1" if cfg.get("wing_dynamics", False) else "massless")
    return dict(species=cfg.get("species", "hawk"), physics=physics, wing_dynamics=physics != "massless",
                energy_cost=cfg.get("energy_cost", 0.0), energy_budget=cfg.get("energy_budget", False),
                soaring=cfg.get("soaring", False))


def run_dirs():
    """Every folder that holds checkpoints: `checkpoints/` plus each `runs/<name>/`."""
    out = []
    if (ROOT / "checkpoints").is_dir():
        out.append(("checkpoints", ROOT / "checkpoints"))
    if RUNS_DIR.is_dir():
        for d in sorted(RUNS_DIR.iterdir()):
            if d.is_dir():
                out.append((f"runs/{d.name}", d))
    return out


def log_path(run_dir):
    p = run_dir / "train.log"
    if not p.exists() and run_dir == ROOT / "checkpoints" and (ROOT / "train.log").exists():
        return ROOT / "train.log"  # the original run logged next to train.py
    return p


def parse_log(path):
    pts = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return pts
    for line in text.splitlines():
        m = LOG_RE.search(line)
        if m:
            it, sps, ret, ln, rings, mx, rr, std, lr, power = m.groups()
            pts.append(dict(it=int(it), sps=float(sps), ret=float(ret), len=float(ln), rings=float(rings),
                            max=float(mx), ring_r=float(rr), std=float(std), lr=float(lr),
                            power=float(power) if power else None))
    return pts


def ckpt_sort_key(p):
    m = re.match(r"iter_(\d+)$", p.stem)
    return (0, int(m.group(1))) if m else (1, p.stem)


def list_runs():
    runs = []
    for name, d in run_dirs():
        cks = sorted(d.glob("*.pt"), key=ckpt_sort_key)
        pts = parse_log(log_path(d))
        runs.append(dict(
            name=name, **run_config(d),
            checkpoints=[dict(path=str(c.relative_to(ROOT)), name=c.name, size=c.stat().st_size,
                              mtime=c.stat().st_mtime) for c in cks],
            last=pts[-1] if pts else None,
            best_rings=max((p["rings"] for p in pts), default=None),
            mtime=max([c.stat().st_mtime for c in cks] + [0]),
        ))
    return runs


def list_replays():
    if not REPLAY_DIR.is_dir():
        return []
    files = sorted(REPLAY_DIR.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [dict(name=f.name, mtime=f.stat().st_mtime, size=f.stat().st_size) for f in files]


def safe_ckpt(rel):
    p = (ROOT / rel).resolve()
    if p.suffix != ".pt" or not p.is_file() or ROOT not in p.parents:
        raise ValueError(f"not a checkpoint in this project: {rel}")
    return p


# --------------------------------------------------------------------- jobs
class Job:
    _next = 1

    def __init__(self, kind, title, cmd, log_file=None, meta=None):
        self.id = Job._next; Job._next += 1
        self.kind, self.title, self.cmd, self.meta = kind, title, cmd, meta or {}
        self.lines = deque(maxlen=400)
        self.status, self.code = "running", None
        self.started, self.ended = time.time(), None
        self.log_file = log_file
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        group = (dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP) if WINDOWS
                 else dict(start_new_session=True))
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1, env=env, **group)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        fh = open(self.log_file, "a", encoding="utf-8") if self.log_file else None
        try:
            for line in self.proc.stdout:
                self.lines.append(line.rstrip("\n"))
                if fh:
                    fh.write(line); fh.flush()
        finally:
            self.code = self.proc.wait()
            if self.status == "running":
                self.status = "done" if self.code == 0 else "failed"
            self.ended = time.time()
            if fh:
                fh.close()

    def stop(self):
        if self.proc.poll() is None:
            self.status = "stopped"
            try:
                if WINDOWS:
                    self.proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(self.proc.pid, signal.SIGINT)
            except (ProcessLookupError, OSError):
                pass
            threading.Timer(8, self._kill).start()

    def _kill(self):
        if self.proc.poll() is None:
            try:
                if WINDOWS:
                    self.proc.kill()
                else:
                    os.killpg(self.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

    def info(self, full=False):
        d = dict(id=self.id, kind=self.kind, title=self.title, status=self.status, code=self.code,
                 started=self.started, ended=self.ended, meta=self.meta, cmd=" ".join(self.cmd[1:]),
                 tail=list(self.lines)[-(400 if full else 6):])
        return d


JOBS = []
LOCK = threading.Lock()
ENV_INFO = dict(status="checking")


def check_env():
    code = ("import json,torch;d='cuda' if torch.cuda.is_available() else "
            "('mps' if torch.backends.mps.is_available() else 'cpu');"
            "print(json.dumps(dict(torch=torch.__version__,device=d)))")
    try:
        r = subprocess.run([PY, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            ENV_INFO.update(json.loads(r.stdout.strip().splitlines()[-1]), status="ok")
        else:
            ENV_INFO.update(status="missing", error="PyTorch is not installed for " + PY)
    except Exception as e:  # noqa: BLE001
        ENV_INFO.update(status="missing", error=str(e))
    ENV_INFO["python"] = PY


def start_training(body):
    with LOCK:
        if any(j.kind == "train" and j.status == "running" for j in JOBS):
            raise ValueError("a training run is already going; stop it first")
    name = body.get("run", "").strip() or time.strftime("run_%Y%m%d_%H%M%S")
    if not NAME_RE.match(name):
        raise ValueError("run name may only use letters, numbers, '-', '_' and '.' (not at the start)")
    out = RUNS_DIR / name
    resume_dir = safe_ckpt(body["resume"]).parent if body.get("resume") else None
    if out.is_dir() and any(out.glob("*.pt")) and resume_dir != out.resolve():
        raise ValueError(f"runs/{name} already has checkpoints; pick another name, "
                         f"or use Continue training on one of its checkpoints")
    out.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "train.py", "--out", str(out.relative_to(ROOT))]
    for key, flag, typ in [("envs", "--envs", int), ("iters", "--iters", int), ("steps", "--steps", int),
                           ("lr", "--lr", float)]:
        if str(body.get(key, "")).strip():
            cmd += [flag, str(typ(body[key]))]
    dev = body.get("device", "auto")
    if dev not in ("auto", "mps", "cuda", "cpu"):
        raise ValueError("bad device")
    cmd += ["--device", dev]
    snaps = str(body.get("snapshots", "")).strip()
    if snaps:
        if not re.fullmatch(r"\d+(\s*,\s*\d+)*", snaps):
            raise ValueError("snapshots should look like 0,10,50,100")
        cmd += ["--snapshots", snaps.replace(" ", "")]
    if body.get("resume"):
        cmd += ["--resume", str(safe_ckpt(body["resume"]).relative_to(ROOT))]
    else:
        species = body.get("species", "hawk")
        if species not in SPECIES:
            raise ValueError(f"unknown species {species}")
        cmd += ["--species", species]
        physics = body.get("physics", "wings-v2")
        if body.get("massless_wings"):
            physics = "massless"
        if physics not in ("wings-v2", "wings-v1", "massless"):
            raise ValueError("bad physics")
        cmd += ["--physics", physics]
        if physics != "massless" and str(body.get("energy_cost", "")).strip():
            cmd += ["--energy-cost", str(float(body["energy_cost"]))]
        if physics == "wings-v2":
            if body.get("energy_budget"):
                cmd += ["--energy-budget"]
            if body.get("soaring"):
                cmd += ["--soaring"]
    if body.get("no_compile"):
        cmd += ["--no-compile"]
    job = Job("train", f"Training runs/{name}" + ("" if body.get("resume") else f" ({body.get('species', 'hawk')})"), cmd, log_file=out / "train.log", meta=dict(run=f"runs/{name}"))
    with LOCK:
        JOBS.append(job)
    if body.get("live", True):
        LIVE.start(out)
    return job


class LiveView:
    """Runs live.py for one run and fans its JSON-line stream out to browsers (server-sent events).
    A browser that connects mid-flight first gets the current flight so far."""
    MAX_FRAMES = 3000

    def __init__(self):
        self.lock = threading.Lock()
        self.proc, self.run, self.subs = None, None, []
        self._clear()

    def _clear(self):
        self.meta = self.meta_line = self.reset = self.waiting = None
        self.events, self.frames, self.dropped = [], deque(), 0
        self.errors = deque(maxlen=20)

    def start(self, run_dir):
        self.stop(quiet=True)
        run_dir = Path(run_dir)
        with self.lock:
            self._clear()
            self.run = run_dir.relative_to(ROOT).as_posix()
            group = (dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP) if WINDOWS else dict(start_new_session=True))
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            self.proc = subprocess.Popen([PY, "live.py", "--run", self.run], cwd=ROOT, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                         bufsize=1, env=env, **group)
            proc = self.proc
        self._broadcast(json.dumps(dict(t="wait", text="Starting the live view…")))
        threading.Thread(target=self._pump, args=(proc,), daemon=True).start()

    def stop(self, quiet=False):
        with self.lock:
            proc, self.proc, self.run = self.proc, None, None
        if proc and proc.poll() is None:
            proc.kill()
        if not quiet:
            self._broadcast(json.dumps(dict(t="stopped", text="Live view is off. Start it from the hub.")))

    def status(self):
        with self.lock:
            running = self.proc is not None and self.proc.poll() is None
            return dict(run=self.run if running else None, iter=(self.meta or {}).get("iter") if running else None)

    def _pump(self, proc):
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                self.errors.append(line)                        # warnings / tracebacks from live.py
                continue
            kind = msg.get("t")
            with self.lock:
                if proc is not self.proc:
                    return
                if kind == "meta":
                    self.meta, self.meta_line = msg["meta"], line
                elif kind == "wait":
                    self.waiting = line
                elif kind == "reset":
                    self.reset, self.events, self.frames, self.dropped = msg, [], deque(), 0
                elif kind == "f":
                    if len(self.frames) >= self.MAX_FRAMES:
                        self.frames.popleft(); self.dropped += 1
                    self.frames.append(line)
                elif kind in ("ring", "th", "end"):
                    self.events.append(line)
            self._broadcast(line)
        proc.wait()
        with self.lock:
            crashed = proc is self.proc
            if crashed:
                self.proc, self.run = None, None
        if crashed:
            why = self.errors[-1] if self.errors else f"exit code {proc.returncode}"
            self._broadcast(json.dumps(dict(t="stopped", text=f"Live view stopped: {why}")))

    def _broadcast(self, line):
        with self.lock:
            subs = list(self.subs)
        for q in subs:
            try:
                q.put_nowait(line)
            except queue.Full:
                pass                                            # a stalled browser just misses frames

    def subscribe(self):
        q = queue.Queue(maxsize=2000)
        with self.lock:
            backlog = []
            if self.meta_line is not None:
                backlog.append(self.meta_line)
                if self.reset is not None:
                    backlog.append(json.dumps(dict(self.reset, k=self.reset["k"] + self.dropped)))
                    backlog += self.events + list(self.frames)
            elif self.proc is not None:
                backlog.append(self.waiting or json.dumps(dict(t="wait", text="Starting the live view…")))
            else:
                backlog.append(json.dumps(dict(t="stopped", text="Live view is off. Start it from the hub.")))
            self.subs.append(q)
        return q, backlog

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subs:
                self.subs.remove(q)


LIVE = LiveView()


def start_replay(body):
    ck = safe_ckpt(body.get("ckpt", ""))
    birds = max(1, min(64, int(body.get("birds", 4))))
    seconds = max(1.0, min(600.0, float(body.get("seconds", 60))))
    seed = int(body.get("seed", 1))
    REPLAY_DIR.mkdir(exist_ok=True)
    rel = ck.relative_to(ROOT)
    stem = "_".join(rel.with_suffix("").parts[-2:])
    out = REPLAY_DIR / f"{stem}_b{birds}_s{seed}.html"
    cmd = [PY, "replay.py", "--ckpt", str(rel), "--birds", str(birds), "--seconds", str(seconds),
           "--seed", str(seed), "--out", str(out.relative_to(ROOT))]
    job = Job("replay", f"Replay of {rel}", cmd, meta=dict(out=out.name))
    with LOCK:
        JOBS.append(job)
    return job


# --------------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if "text" in ctype or "json" in ctype else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, (ROOT / "hub.html").read_bytes(), "text/html")
            if u.path == "/api/state":
                with LOCK:
                    jobs = [j.info() for j in reversed(JOBS)]
                return self._send(200, dict(env=ENV_INFO, jobs=jobs, runs=list_runs(), replays=list_replays(),
                                            species=SPECIES, live=LIVE.status()))
            if u.path == "/live":
                page = (ROOT / "viewer_template.html").read_text(encoding="utf-8")
                page = page.replace("__DATA__", json.dumps(dict(live=True, stream="/api/live/stream")))
                return self._send(200, page, "text/html")
            if u.path == "/api/live/stream":
                return self._stream()
            if u.path == "/api/metrics":
                name = q.get("run", [""])[0]
                d = dict(run_dirs()).get(name)
                if d is None:
                    return self._send(404, dict(error="unknown run"))
                return self._send(200, dict(run=name, points=parse_log(log_path(d))))
            if u.path == "/api/job":
                jid = int(q.get("id", ["0"])[0])
                job = next((j for j in JOBS if j.id == jid), None)
                return self._send(200, job.info(full=True)) if job else self._send(404, dict(error="no such job"))
            if u.path.startswith("/replays/"):
                name = unquote(u.path[len("/replays/"):])
                f = (REPLAY_DIR / name).resolve()
                if f.parent == REPLAY_DIR.resolve() and f.suffix == ".html" and f.is_file():
                    return self._send(200, f.read_bytes(), "text/html")
            return self._send(404, dict(error="not found"))
        except Exception as e:  # noqa: BLE001
            return self._send(500, dict(error=str(e)))

    def _stream(self):
        q, backlog = LIVE.subscribe()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write("".join(f"data: {x}\n\n" for x in backlog).encode())
            self.wfile.flush()
            while True:
                try:
                    chunk = [q.get(timeout=15)]
                    while not q.empty() and len(chunk) < 200:
                        chunk.append(q.get_nowait())
                    self.wfile.write("".join(f"data: {x}\n\n" for x in chunk).encode())
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            LIVE.unsubscribe(q)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if u.path == "/api/train":
                return self._send(200, start_training(body).info())
            if u.path == "/api/live/start":
                d = dict(run_dirs()).get(body.get("run", ""))
                if d is None:
                    raise ValueError("unknown run")
                LIVE.start(d)
                return self._send(200, LIVE.status())
            if u.path == "/api/live/stop":
                LIVE.stop()
                return self._send(200, dict(ok=True))
            if u.path == "/api/replay":
                return self._send(200, start_replay(body).info())
            if u.path == "/api/stop":
                job = next((j for j in JOBS if j.id == int(body.get("id", 0))), None)
                if job:
                    job.stop()
                return self._send(200, dict(ok=True))
            if u.path == "/api/delete_replay":
                f = (REPLAY_DIR / body.get("name", "")).resolve()
                if f.parent == REPLAY_DIR.resolve() and f.suffix == ".html" and f.is_file():
                    f.unlink()
                return self._send(200, dict(ok=True))
            return self._send(404, dict(error="not found"))
        except ValueError as e:
            return self._send(400, dict(error=str(e)))
        except Exception as e:  # noqa: BLE001
            return self._send(500, dict(error=str(e)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    threading.Thread(target=check_env, daemon=True).start()
    ThreadingHTTPServer.daemon_threads = True
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Neural Flight Hub running at {url}  (Ctrl+C to quit)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for j in JOBS:
            j.stop()
        LIVE.stop(quiet=True)
        srv.server_close()


if __name__ == "__main__":
    main()
