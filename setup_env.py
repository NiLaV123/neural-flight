"""
Install the requirements into the current Python, picking the PyTorch build that uses your GPU.
start.bat runs this on every start, so a CPU-only PyTorch is replaced automatically.

    .venv\\Scripts\\python setup_env.py            # install (first run)
    .venv\\Scripts\\python setup_env.py --ensure   # check; reinstall only if the NVIDIA GPU isn't usable

Why it is needed: on Windows, a plain `pip install torch` gives a CPU-only build, and RTX 50-series
cards (Blackwell, compute capability 12.x) only work with PyTorch built for CUDA 12.8 or newer.
"""
import re
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

# PyTorch CUDA builds, newest first: (tag, lowest driver CUDA version it needs)
CUDA_BUILDS = [("cu132", (13, 2)), ("cu130", (13, 0)), ("cu128", (12, 8)), ("cu126", (12, 6))]
# Windows Triton minor versions supported by each PyTorch release.
TRITON_MINOR = {4: 1, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6, 11: 6, 12: 7, 13: 7, 14: 8, 15: 9}


def nvidia():
    """(driver CUDA version, GPU name, compute capability) or None when there is no NVIDIA GPU."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    # older drivers print "CUDA Version: 12.8", newer ones "CUDA UMD Version: 13.5"
    m = re.search(r"CUDA\s+(?:UMD\s+)?Version:\s*(\d+)\.(\d+)", out)
    if m:
        driver = (int(m.group(1)), int(m.group(2)))
    elif q:
        driver = (99, 0)        # GPU present but the format is new to us: try the newest builds first
    else:
        return None
    name, cap = (q[0].rsplit(",", 1) + ["0"])[:2] if q else ("NVIDIA GPU", "0")
    try:
        cap = tuple(int(x) for x in cap.strip().split("."))
    except ValueError:
        cap = (0, 0)
    return driver, name.strip(), cap


def pick_builds(driver, cap):
    """CUDA builds this driver and card can use, newest first. RTX 50 (compute 12.x) needs cu128+:
    a cu126 build would install but not run on it."""
    return [tag for tag, need in CUDA_BUILDS if driver >= need and not (cap >= (12, 0) and tag == "cu126")]


def gpu_works():
    """Ask a fresh Python whether the installed PyTorch can actually run a kernel on the GPU."""
    code = ("import torch,sys\n"
            "ok = torch.cuda.is_available()\n"
            "if ok:\n"
            "    try:\n"
            "        x = torch.ones(64, 64, device='cuda'); (x @ x).sum().item()\n"
            "    except Exception as e:\n"
            "        print('GPU kernel failed:', str(e).splitlines()[0]); ok = False\n"
            "print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu ok' if ok else 'gpu NOT usable')\n"
            "sys.exit(0 if ok else 1)\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    msg = (r.stdout.strip().splitlines() or [r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "torch missing"])[-1]
    return r.returncode == 0, msg


def ensure_triton():
    """Use the Windows Triton release series matched to the installed PyTorch."""
    try:
        torch_version = version("torch")
    except PackageNotFoundError:
        return
    match = re.match(r"2\.(\d+)\.", torch_version)
    minor = TRITON_MINOR.get(int(match[1])) if match else None
    if minor is None:
        print(f"No automatic Triton version selection for PyTorch {torch_version}.")
        return
    r = subprocess.run([sys.executable, "-c", "import triton; print(triton.__version__)"],
                       capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip().split(".")[:2] == ["3", str(minor)]:
        return
    print(f"Installing Triton 3.{minor} for PyTorch {torch_version}...")
    package = f"triton-windows>=3.{minor},<3.{minor + 1}"
    if subprocess.call([sys.executable, "-m", "pip", "install", package]) != 0:
        print("Could not install triton-windows; training will run without torch.compile (slower).")


def install(gpu):
    pip = [sys.executable, "-m", "pip", "install"]
    if gpu and sys.platform == "win32":
        driver, name, cap = gpu
        tags = pick_builds(driver, cap)
        if not tags:
            print(f"Your NVIDIA driver supports CUDA {driver[0]}.{driver[1]}, which is too old for "
                  f"{'an RTX 50-series card (needs 12.8+)' if cap >= (12, 0) else 'PyTorch (needs 12.6+)'}.\n"
                  "Update the driver from https://www.nvidia.com/drivers and run start.bat again. "
                  "Installing the CPU build for now.")
            subprocess.check_call(pip + ["torch"])
        else:
            for i, tag in enumerate(tags):                  # newest first; older if no wheel for this Python
                print(f"{name} found (driver CUDA {driver[0]}.{driver[1]}): installing PyTorch for {tag}")
                r = subprocess.call(pip + ["--upgrade", "--force-reinstall", "torch",
                                           "--index-url", f"https://download.pytorch.org/whl/{tag}"])
                if r == 0:
                    break
                if i == len(tags) - 1:
                    raise SystemExit("Could not install a GPU build of PyTorch for this Python version. "
                                     "Python 3.12 or 3.13 from python.org is the safest choice.")
                print(f"No {tag} build for this Python; trying an older CUDA build.")
    elif sys.platform == "win32":
        print("No NVIDIA GPU found: installing the CPU build of PyTorch (training will be slow).")
        subprocess.check_call(pip + ["torch"])
    else:
        subprocess.check_call(pip + ["torch"])          # Linux wheels include CUDA; macOS uses Apple GPU
    subprocess.check_call(pip + ["numpy"])


def main():
    ensure = "--ensure" in sys.argv
    gpu = nvidia()
    if ensure:
        if gpu is None:
            return                                          # nothing to fix without an NVIDIA GPU
        ok, msg = gpu_works()
        if ok:
            print(f"GPU ready: {gpu[1]} ({msg})")
            if sys.platform == "win32":
                ensure_triton()
            return
        print(f"{gpu[1]} found but PyTorch can't use it ({msg}). Reinstalling PyTorch with GPU support...")
    install(gpu)
    if gpu:
        ok, msg = gpu_works()
        print(("GPU ready: " if ok else "Warning, GPU still not usable: ") + f"{gpu[1]} ({msg})")
        if ok and sys.platform == "win32":
            ensure_triton()
    print("Done.")


if __name__ == "__main__":
    main()
