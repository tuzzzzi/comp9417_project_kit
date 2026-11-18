"""Rebuild the two selected Stage 1 artifacts, without rerunning model selection.

This is A's frozen 2010-2013 training helper, not the final dual-task train.py.
"""

import argparse
from pathlib import Path
import sys

import joblib
from threadpoolctl import threadpool_limits

from harness import dayahead as da
from .demand_models import build_models
from .run_demand_study import DEFAULT_DATA
from .stage1_data import ROOT, VALIDATION_START, load_stage1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Use the course Python 3.11 environment")
    features, truth = load_stage1(args.data)
    train = da.market_day(features.index) < VALIDATION_START
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=2):
        for name, model in build_models().items():
            model.fit(features.loc[train], truth.nsw_demand_mw.loc[train])
            path = args.output / f"demand_{name}_frozen.joblib"
            joblib.dump(model, path, compress=3)
            print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
