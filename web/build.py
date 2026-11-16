"""Build a single self-contained HTML: wasm sim + policy snapshots + WebGPU renderer.

    python web/build.py --run runs/fresh          # writes docs/index.html
"""
import argparse, base64, glob, json, os, re, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--run", default="runs/fresh")
p.add_argument("--out", default=os.path.join(here, "..", "docs", "index.html"))
a = p.parse_args()

subprocess.run(["cargo", "build", "--release", "--target", "wasm32-unknown-unknown", "-q"], cwd=os.path.join(here, "sim"), check=True)
wasm = open(os.path.join(here, "sim/target/wasm32-unknown-unknown/release/birdsim.wasm"), "rb").read()

ckpts = sorted(glob.glob(os.path.join(a.run, "iter_*.pt")), key=lambda s: int(re.findall(r"iter_(\d+)", s)[0]))
pol_json = os.path.join(here, "build/policies.json")
subprocess.run([sys.executable, os.path.join(here, "export_policy.py"), *ckpts, "--out", pol_json], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
pols = json.load(open(pol_json))

# annotate each snapshot with its training stats from the log
log = {}
logf = os.path.join(a.run, "train.log")
if os.path.exists(logf):
    for line in open(logf):
        m = re.match(r"it\s+(\d+) .*rings/ep\s+([\d.]+)", line)
        if m: log[int(m.group(1))] = float(m.group(2))
total_steps = lambda it: it * 8192 * 32
for pol in pols:
    it = pol["iter"]
    seen = f"{total_steps(it) / 1e6:.0f}M sim steps" if it else "untrained"
    r = log.get(it)
    pol["note"] = f"{seen}" + (f" · {r:.1f} rings / 40 s in training" if r is not None and it else " · random weights")

html = open(os.path.join(here, "index_template.html")).read()
html = html.replace("__WASM__", base64.b64encode(wasm).decode()).replace("__POLICIES__", json.dumps(pols))
open(a.out, "w").write(html)
print(f"wasm {len(wasm)/1024:.0f} KB, {len(pols)} brains, html {len(html)/1e6:.2f} MB -> {a.out}")
