"""One entry point for the integrated A/B/C/D experiments.

Default: verify existing Stage 2 results, with no training.
Use --list to inspect stages. Only explicitly selected stages are run.
The EDA notebook remains the C entry point; no report or final package is generated.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from .stage1_data import ROOT, DATA_PATH

STEPS = {
    'validate': ('experiments.validate_models', 'A/B: fit the four fixed candidates; evaluate on 2014'),
    'demand-check': ('experiments.check_demand_handoff', 'A: verify supplied frozen models and results'),
    'frozen': ('experiments.backtest_frozen', '2015–2026: evaluate all four frozen candidates'),
    'adaptation': ('experiments.backtest_adaptation', 'D: simulate three strategies; this retrains historical models'),
    'results': ('experiments.adaptation_results', 'D: regenerate strategy tables and figures without fitting'),
    'verify': ('experiments.verify_adaptation', 'D: verify saved strategy results without fitting'),
}


def plan(steps, data):
    """Build commands without launching any work; used by --dry-run."""
    commands = []
    for step in steps:
        command = [sys.executable, '-m', STEPS[step][0]]
        if step in {'validate', 'demand-check', 'frozen', 'adaptation'}:
            command += ['--data', str(data)]
        if step == 'validate':
            # Preserve teammate evidence in outputs/stage1 when reproducing models.
            command += ['--output', str(ROOT / 'outputs/reproduction')]
        commands.append(command)
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', nargs='+', choices=STEPS, default=['verify'],
                        help='Explicit stages in execution order; default: verify')
    parser.add_argument('--data', type=Path, default=DATA_PATH)
    parser.add_argument('--list', action='store_true', help='Describe stages and exit')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    args = parser.parse_args()
    if args.list:
        for name, (_, meaning) in STEPS.items():
            print(f'{name:14s} {meaning}')
        return
    if sys.version_info[:2] != (3, 11):
        parser.error('Use the course Python 3.11 environment.')
    if len(args.steps) != len(set(args.steps)):
        parser.error('Do not repeat a stage in one invocation.')
    order = [list(STEPS).index(step) for step in args.steps]
    if order != sorted(order):
        parser.error('Use stage order: validate, demand-check, frozen, adaptation, results, verify.')
    data = args.data.expanduser().resolve()
    commands = plan(args.steps, data)
    if args.dry_run:
        for command in commands:
            print(subprocess.list2cmdline(command))
        return
    if any(step in {'validate', 'demand-check', 'frozen', 'adaptation'} for step in args.steps):
        if not data.is_file():
            parser.error(f'Dataset missing: {data}')
    # Validate dependencies before launching an expensive stage. Later selected stages
    # can use files produced by earlier ones; nothing is retrained implicitly.
    produced = set()
    for step in args.steps:
        required = []
        if step in {'frozen', 'adaptation'}:
            required += [ROOT / 'day_types.csv', ROOT / 'artifacts/demand_mlp_frozen.joblib']
        if step == 'frozen':
            required += [ROOT / 'artifacts/demand_ridge_frozen.joblib']
        if step == 'adaptation' and 'frozen' not in produced:
            required += [ROOT / 'outputs/frozen_backtest' / name for name in
                         ['price_hist_gbc_frozen.joblib', 'demand_predictions.csv.gz', 'price_predictions.csv.gz']]
        if step in {'results', 'verify'} and 'adaptation' not in produced:
            required += [ROOT / 'outputs/adaptation' / name for name in
                         ['retraining.csv', 'detector_daily.csv', 'demand_predictions.csv.gz', 'price_predictions.csv.gz']]
        if step == 'verify' and not {'adaptation', 'results'} & produced:
            required += [ROOT / 'outputs/adaptation/monthly_metrics.csv', ROOT / 'outputs/adaptation/yearly_metrics.csv']
        # A new simulation must have its summaries regenerated before verification.
        if step == 'verify' and 'adaptation' in produced and 'results' not in produced:
            parser.error('Run results after adaptation and before verify.')
        missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
        if missing:
            parser.error(f'{step} requires: {", ".join(missing)}')
        produced.add(step)
    env = os.environ.copy()
    env.setdefault('LOKY_MAX_CPU_COUNT', '2')
    for step, command in zip(args.steps, commands):
        print(f'\n[{step}] {STEPS[step][1]}', flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
