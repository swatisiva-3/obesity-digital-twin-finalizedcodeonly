"""
00_setup_environment.py
========================
One-time bootstrap: creates venv/ and installs requirements.txt into it.

Run: python 00_setup_environment.py
Then, every session:
    source venv/bin/activate        (Windows: venv\\Scripts\\activate)
"""
import subprocess
import sys
import venv
from pathlib import Path

VENV_DIR = Path(__file__).parent / "venv"


def main():
    if VENV_DIR.exists():
        print(f"venv already exists at {VENV_DIR} -- skipping creation. "
              f"Delete it first if you want a clean reinstall.")
    else:
        print(f"Creating virtual environment at {VENV_DIR} ...")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)

    pip = VENV_DIR / ("Scripts/pip.exe" if sys.platform == "win32" else "bin/pip")
    req = Path(__file__).parent / "requirements.txt"
    print(f"Installing {req} ...")
    subprocess.check_call([str(pip), "install", "--upgrade", "pip"])
    subprocess.check_call([str(pip), "install", "-r", str(req)])
    print("\nDone. Activate with:")
    if sys.platform == "win32":
        print(r"    venv\Scripts\activate")
    else:
        print("    source venv/bin/activate")


if __name__ == "__main__":
    main()
