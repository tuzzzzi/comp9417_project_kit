"""One-command reproduction of A, including a fresh group-validator fit."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from .run_demand_study import DEFAULT_DATA
from .stage1_data import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    data = str(args.data.resolve())
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
    handoff_dir = ROOT / "handoffs" / "A_demand"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    with (handoff_dir / "environment.freeze.txt").open("w") as stream:
        subprocess.run([sys.executable, "-m", "pip", "freeze", "--disable-pip-version-check"],
                       stdout=stream, check=True)
    steps = [
        ["experiments.run_demand_study", "--data", data],
        ["experiments.validate_models", "--task", "demand", "--data", data,
         "--output", str(ROOT / "outputs/reproduction")],
        ["experiments.plot_demand_results"],
        ["experiments.check_demand_handoff", "--data", data,
         "--reproduced", str(ROOT / "outputs/reproduction")],
        ["experiments.write_demand_handoff"],
        ["experiments.package_demand_handoff"],
    ]
    for step in steps:
        print("RUN", " ".join(step), flush=True)
        subprocess.run([sys.executable, "-m", *step], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
