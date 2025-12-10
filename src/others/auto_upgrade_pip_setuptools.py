#!.venv/Scripts/python

import subprocess
import sys

def update_pip_setuptools():
    """Main function"""
    subprocess.check_call([
        sys.executable, "-m",
        "pip", "install", "--upgrade", "pip", "setuptools"
    ])

if __name__ == "__main__":
    # Optional: Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Upgrading pip and setuptools in virtual environment...")
        update_pip_setuptools()
    else:
        print("Not in a virtual environment. Skipping update.")
