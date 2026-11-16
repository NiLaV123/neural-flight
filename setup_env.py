"""
Install the requirements into the current Python, picking the right PyTorch build.
Used by start.bat on Windows, where plain `pip install torch` gives a CPU-only build.

    .venv\\Scripts\\python setup_env.py
"""
import re
import shutil
import subprocess
import sys


def driver_cuda():
    """CUDA version the installed NVIDIA driver supports, e.g. (13, 0), or None without a GPU."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def main():
    pip = [sys.executable, "-m", "pip", "install"]
    cmd = pip + ["torch"]
    cuda = driver_cuda()
    if sys.platform == "win32" and cuda:
        # cu130 covers the newest cards (RTX 50xx); cu126 works with older drivers.
        tag = "cu130" if cuda >= (13, 0) else "cu126"
        if cuda < (12, 6):
            print(f"Your NVIDIA driver supports CUDA {cuda[0]}.{cuda[1]}; PyTorch needs 12.6 or newer. "
                  "Update the driver from https://www.nvidia.com/drivers if training falls back to the CPU.")
        print(f"NVIDIA GPU found (driver CUDA {cuda[0]}.{cuda[1]}): installing PyTorch for {tag}")
        cmd += ["--index-url", f"https://download.pytorch.org/whl/{tag}"]
    elif sys.platform == "win32":
        print("No NVIDIA GPU found: installing the CPU build of PyTorch (training will be slow).")
    subprocess.check_call(cmd)
    subprocess.check_call(pip + ["numpy"])
    print("Done.")


if __name__ == "__main__":
    main()
