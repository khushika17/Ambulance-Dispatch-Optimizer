"""Convenience launcher.

Usage:
    python run.py train      # train (or retrain) the pipeline
    python run.py serve      # start the Flask web app
    python run.py all        # train then serve
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(cmd: list[str]) -> int:
    print(f"\n> {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"

    if cmd == "train":
        return run([PY, "backend/train.py"])

    if cmd == "serve":
        return run([PY, "backend/app.py"])

    if cmd == "all":
        rc = run([PY, "backend/train.py"])
        if rc != 0:
            return rc
        return run([PY, "backend/app.py"])

    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
