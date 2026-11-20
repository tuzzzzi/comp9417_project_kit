"""Rebuild both final artifacts solely from the released CSV.

Usage: python train.py /path/to/nem_weather_halfhourly_2010_2026.csv
The package is independent of experiments/, notebooks and historical outputs.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from demand import build_model as demand_model
from price import build_model as price_model
from features import WEATHER_COLS, build_features, market_day

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output', type=Path, default=HERE / 'artifacts')
    args = parser.parse_args()
    settings = json.loads((HERE / 'settings.json').read_text())
    if platform.python_version_tuple()[:2] != ('3', '11'):
        raise RuntimeError('Python 3.11 is required')
    for library, expected in settings['versions'].items():
        actual = importlib.metadata.version(library)
        if actual != expected:
            raise RuntimeError(f'{library}: expected {expected}, found {actual}')
    started = time.perf_counter()
    dataset_hash = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    if dataset_hash != settings['released_csv_sha256']:
        raise ValueError('CSV differs from the fixed course dataset')
    df = pd.read_csv(args.csv, parse_dates=['timestamp']).set_index('timestamp')
    expected = pd.date_range('2010-01-01 00:30', '2026-11-01 00:00', freq='30min')
    if not df.index.equals(expected) or not np.isfinite(df.to_numpy()).all():
        raise ValueError('Expected the complete released half-hour dataset')
    x = build_features(df, df[WEATHER_COLS])
    days = market_day(x.index)
    mask = (days >= pd.Timestamp(settings['first_market_day'])) & (days <= pd.Timestamp(settings['last_market_day']))
    x = x.loc[mask]
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Nonfinite training features')
    y_demand = df.nsw_demand.loc[x.index]
    y_price = (df.nsw_price.loc[x.index] > x.previous_price_mean).astype(int)
    models = {'demand': demand_model(settings['demand_params']),
              'price': price_model(settings['price_params'])}
    args.output.mkdir(parents=True, exist_ok=True)
    fit_warnings = []
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for name, target in [('demand', y_demand), ('price', y_price)]:
            print(f'Fitting {name}: {len(x)} half-hours', flush=True)
            models[name].fit(x, target)
            joblib.dump(models[name], args.output / f'{name}.joblib', compress=3)
        fit_warnings = [str(w.message) for w in caught]
    record = {'first_market_day': settings['first_market_day'],
              'last_market_day': settings['last_market_day'], 'n_halfhours': len(x),
              'feature_names': list(x.columns), 'released_csv_sha256': dataset_hash,
              'versions': settings['versions'], 'python': platform.python_version(),
              'strategy': settings['strategy'], 'fit_warnings': fit_warnings,
              'fitted_state_hashes': {name: joblib.hash(m) for name,m in models.items()},
              'artifact_sha256': {name: hashlib.sha256((args.output/f'{name}.joblib').read_bytes()).hexdigest() for name in models}}
    (args.output / 'training.json').write_text(json.dumps(record, indent=2)+'\n')
    print(f'Artifacts rebuilt in {time.perf_counter()-started:.1f} seconds', flush=True)
    for warning in fit_warnings:
        print('Training warning:', warning, flush=True)


if __name__ == '__main__':
    main()
