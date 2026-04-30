from __future__ import annotations

from dvgrpsig.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["run_flow", *(__import__("sys").argv[1:])]))
