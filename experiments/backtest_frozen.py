"""Frozen Stage 2: causal daily forecasts, official metrics and day-block intervals.

Run from the kit root: python -m experiments.backtest_frozen
No model selection, adaptation or clipping is performed on historical test data.
"""
import argparse
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from harness import dayahead as da
from . import price_models
from .stage1_data import ROOT, DATA_PATH, load_stage1, build_features, baseline_predictions

DEMAND_MODELS = ('yesterday', 'last_week', 'ridge', 'mlp')
PRICE_MODELS = ('yesterday', 'last_week', 'logistic', 'hist_gbc')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric(task, y, p):
    if task == 'demand':
        return {'MAPE': da.mape(y, p), 'MAE': da.mae(y, p)}
    return {'accuracy': da.accuracy(y, p), 'balanced_accuracy': da.balanced_accuracy(y, p)}


def intervals(y, p, task, rng, n_boot):
    # Each day has exactly 48 half-hours. Resampling daily means is algebraically
    # identical to resampling whole days and averaging all their half-hours.
    errors = 100*np.abs(y-p)/np.abs(y) if task == 'demand' else 100*(y == p).astype(float)
    daily = errors.reshape(-1, 48).mean(axis=1)
    estimate = errors.mean()
    halfwidth = 1.96*errors.std(ddof=1)/np.sqrt(len(errors))
    iid = [estimate-halfwidth, estimate+halfwidth]
    if task == 'price':
        iid = np.clip(iid, 0, 100)
    draws = rng.integers(0, len(daily), size=(n_boot, len(daily)))
    block = np.percentile(daily[draws].mean(axis=1), [2.5, 97.5])
    return {'ci_metric': 'MAPE' if task == 'demand' else 'accuracy',
            'iid_lower': float(iid[0]), 'iid_upper': float(iid[1]),
            'day_bootstrap_lower': float(block[0]), 'day_bootstrap_upper': float(block[1]),
            'bootstrap_replicates': n_boot}


def aggregate(frame, task, frequency, rng, n_boot):
    names = DEMAND_MODELS if task == 'demand' else PRICE_MODELS
    if frequency == 'daily':
        keys = frame.market_day.dt.strftime('%Y-%m-%d')
    elif frequency == 'monthly':
        keys = frame.market_day.dt.strftime('%Y-%m')
    elif frequency == 'yearly':
        keys = frame.market_day.dt.year.astype(str)
    else:
        raise ValueError(frequency)
    rows = []
    for period, g in frame.groupby(keys, sort=True):
        y = g.actual.to_numpy()
        for name in names:
            p = g[name].to_numpy()
            row = {'task': task, 'model': name, 'period': period,
                   'n_halfhours': len(g), 'n_days': g.market_day.nunique(), **metric(task,y,p)}
            if frequency == 'monthly':
                row.update(intervals(y,p,task,rng,n_boot))
            rows.append(row)
    return pd.DataFrame(rows)


def plot_results(monthly, yearly, output):
    for task, names in [('demand', ('ridge','mlp')), ('price', ('logistic','hist_gbc'))]:
        score = 'MAPE' if task == 'demand' else 'accuracy'
        fig, axes = plt.subplots(2,1,figsize=(12,7),sharex=True)
        for ax, name in zip(axes, names):
            s = monthly.loc[(monthly.task == task) & (monthly.model == name)].copy()
            t = pd.to_datetime(s.period)
            ax.fill_between(t,s.day_bootstrap_lower,s.day_bootstrap_upper,alpha=.24,label='95% whole-day bootstrap')
            ax.fill_between(t,s.iid_lower,s.iid_upper,alpha=.45,label='95% IID normal approximation')
            ax.plot(t,s[score],lw=1,label=f'{name}, frozen after 2013')
            for baseline, style in [('yesterday','--'),('last_week',':')]:
                b=monthly.loc[(monthly.task==task)&(monthly.model==baseline)]
                ax.plot(pd.to_datetime(b.period),b[score],style,lw=.8,alpha=.7,label=baseline)
            ax.set(ylabel=f'{score} (%)',title=f'{name}: monthly performance')
            ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2)
        axes[-1].set_xlabel('Market month (AEST)')
        fig.tight_layout();fig.savefig(output/f'{task}_monthly_intervals.png',dpi=160);plt.close(fig)

        fig,axes=plt.subplots(1,2,figsize=(12,4))
        scores=['MAPE','MAE'] if task=='demand' else ['accuracy','balanced_accuracy']
        for ax,col in zip(axes,scores):
            for name,g in yearly.loc[yearly.task==task].groupby('model'):
                ax.plot(g.period.astype(int),g[col],marker='o',label=name)
            ax.set(xlabel='Market year (2026: Jan–Oct)',ylabel=f'{col} '+('(MW)' if col=='MAE' else '(%)'))
            ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(output/f'{task}_yearly.png',dpi=160);plt.close(fig)


def day_type_analysis(frame, path, output):
    types=pd.read_csv(path,parse_dates=['market_day'])
    assert not types.market_day.duplicated().any()
    rows=[]
    for name in DEMAND_MODELS:
        joined=frame[['market_day','actual',name]].merge(types,on='market_day',how='left',validate='many_to_one')
        assert joined.day_type.notna().all()
        joined['year']=joined.market_day.dt.year
        for (year,kind),g in joined.groupby(['year','day_type']):
            rows.append({'year':year,'day_type':kind,'model':name,'n_days':g.market_day.nunique(),
                         **metric('demand',g.actual.to_numpy(),g[name].to_numpy())})
    result=pd.DataFrame(rows)
    result.to_csv(output/'demand_errors_by_day_type.csv',index=False)
    # This decomposes the yearly mean error exactly into day-type share * within-type
    # error. It is descriptive and does not establish causal effects of day types.
    for name in DEMAND_MODELS:
        s=result.loc[result.model==name]
        for year,g in s.groupby('year'):
            actual=frame.loc[frame.market_day.dt.year==year]
            np.testing.assert_allclose(np.average(g.MAPE,weights=g.n_days),da.mape(actual.actual,actual[name]),atol=1e-10)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,name in zip(axes,['ridge','mlp']):
        for kind,g in result.loc[result.model==name].groupby('day_type'):
            ax.plot(g.year,g.MAPE,'o-',label=f'Type {kind}')
        ax.set(xlabel='Market year (2026: Jan–Oct)',ylabel='MAPE (%)',title=f'{name}: within-type error')
        ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'demand_day_type_errors.png',dpi=160);plt.close(fig)
    days=frame[['market_day']].drop_duplicates().merge(types,on='market_day',validate='one_to_one')
    composition=pd.crosstab(days.market_day.dt.to_period('M'),days.day_type,normalize='index')*100
    composition.index.name='month'
    composition.to_csv(output/'monthly_day_type_shares.csv')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DATA_PATH)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/frozen_backtest')
    parser.add_argument('--day-types',type=Path,default=ROOT/'day_types.csv')
    parser.add_argument('--start',default='2015-01-01')
    parser.add_argument('--end',default='2026-10-31')
    parser.add_argument('--bootstrap',type=int,default=1000)
    args=parser.parse_args()
    assert args.bootstrap>=1000
    start,end=pd.Timestamp(args.start),pd.Timestamp(args.end)
    assert pd.Timestamp('2015-01-01')<=start<=end<=pd.Timestamp('2026-10-31')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    begin=time.perf_counter()
    data=da.load_dataset(args.data)
    expected=pd.date_range('2010-01-01 00:30','2026-11-01 00:00',freq='30min')
    assert data.index.equals(expected) and not data.isna().any().any()
    stage_x,stage_y=load_stage1(args.data)
    train=da.market_day(stage_x.index)<pd.Timestamp('2014-01-01')
    demand={name:joblib.load(ROOT/f'artifacts/demand_{name}_frozen.joblib') for name in ['ridge','mlp']}
    # Snapshot fitted parameters before simulation: no fit() is allowed afterwards.
    prices=price_models.build_models()
    caught=[]
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        for name,model in prices.items():
            model.fit(stage_x.loc[train],stage_y.loc[train,'nsw_price_up'])
            joblib.dump(model,out/f'price_{name}_frozen.joblib',compress=3)
        caught.extend(str(w.message) for w in ws)
    models={**demand,**prices}
    before={name:joblib.hash(model) for name,model in models.items()}
    validation=[]
    with threadpool_limits(limits=2):
        for task,group in [('demand',demand),('price',prices)]:
            y=stage_y.loc[~train,'nsw_demand_mw' if task=='demand' else 'nsw_price_up'].to_numpy()
            for name,model in group.items():
                validation.append({'task':task,'model':name,**metric(task,y,model.predict(stage_x.loc[~train]))})
    pd.DataFrame(validation).to_csv(out/'stage1_recheck.csv',index=False)
    records={'demand':[],'price':[]}
    timing=[]; boundary_checks=[]
    all_days=pd.date_range(start,end)
    print(f'Frozen daily simulation: {len(all_days)} days; no retraining.',flush=True)
    with threadpool_limits(limits=2):
        for i,day in enumerate(all_days):
            pos=int((day-pd.Timestamp('2010-01-01')).days)*48
            # The shared features require only seven completed days. No target-day
            # market values are supplied to feature construction.
            history=data.iloc[pos-7*48:pos]
            weather=data.iloc[pos:pos+48][da.WEATHER_COLS]
            assert da.market_day(history.index).max()<day
            x=build_features(history,weather)
            assert x.columns.equals(stage_x.columns) and np.isfinite(x.to_numpy()).all()
            if i==0 or day.month==1 and day.day==1 or day==end or day==pd.Timestamp('2017-05-14'):
                official_h,official_w=da.make_inputs(data,day)
                pd.testing.assert_frame_equal(x,build_features(official_h,official_w))
                boundary_checks.append(str(day.date()))
            # Only after constructing legal inputs, reveal today's labels for scoring.
            observed=data.iloc[pos:pos+48]
            truth={'demand':observed.nsw_demand.to_numpy(),
                   'price':(observed.nsw_price.to_numpy()>history.nsw_price.iloc[-48:].mean()).astype(int)}
            if str(day.date()) in boundary_checks:
                official_truth=da.targets(data,day)
                for task,col in [('demand','nsw_demand_mw'),('price','nsw_price_up')]:
                    np.testing.assert_array_equal(truth[task],official_truth[col])
                    for lag,name in [(1,'yesterday'),(7,'last_week')]:
                        np.testing.assert_array_equal(baseline_predictions(x,task)[name],da.baseline(history,weather,lag)[col])
            t=time.perf_counter()
            for task,group in [('demand',demand),('price',prices)]:
                result=pd.DataFrame({'timestamp':weather.index,'market_day':day,'actual':truth[task],
                                     **{k:v.to_numpy() for k,v in baseline_predictions(x,task).items()}})
                for name,model in group.items():
                    pred=np.asarray(model.predict(x))
                    assert pred.shape==(48,) and np.isfinite(pred).all()
                    if task=='price':assert np.isin(pred,[0,1]).all()
                    result[name]=pred
                records[task].append(result)
            timing.append({'market_day':day,'four_model_predict_seconds':time.perf_counter()-t})
            if (i+1)%365==0 or i==len(all_days)-1:
                print(f'{i+1}/{len(all_days)} days, through {day.date()}, elapsed {time.perf_counter()-begin:.1f}s',flush=True)
    assert before=={name:joblib.hash(model) for name,model in models.items()}, 'Model state changed during backtest'
    frames={task:pd.concat(parts,ignore_index=True) for task,parts in records.items()}
    monthly=[];yearly=[];daily=[];summary=[]
    rng=np.random.default_rng(9417)
    for task,frame in frames.items():
        assert len(frame)==len(all_days)*48 and not frame.timestamp.duplicated().any()
        assert frame.groupby('market_day').size().eq(48).all()
        frame.to_csv(out/f'{task}_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        monthly.append(aggregate(frame,task,'monthly',rng,args.bootstrap))
        yearly.append(aggregate(frame,task,'yearly',rng,args.bootstrap))
        daily.append(aggregate(frame,task,'daily',rng,args.bootstrap))
        names=DEMAND_MODELS if task=='demand' else PRICE_MODELS
        for period,mask in [('all',np.ones(len(frame),dtype=bool)),('2015–2024',frame.market_day.dt.year<=2024),('2025–2026.10',frame.market_day.dt.year>=2025)]:
            g=frame.loc[mask]
            if g.empty:continue
            for name in names:
                summary.append({'task':task,'model':name,'period':period,'n_days':g.market_day.nunique(),**metric(task,g.actual.to_numpy(),g[name].to_numpy())})
    monthly=pd.concat(monthly,ignore_index=True);yearly=pd.concat(yearly,ignore_index=True)
    monthly.to_csv(out/'monthly_metrics.csv',index=False)
    yearly.to_csv(out/'yearly_metrics.csv',index=False)
    pd.concat(daily,ignore_index=True).to_csv(out/'daily_metrics.csv',index=False)
    summary=pd.DataFrame(summary);summary.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(timing).to_csv(out/'daily_timing.csv',index=False)
    monthly['iid_width']=monthly.iid_upper-monthly.iid_lower
    monthly['block_width']=monthly.day_bootstrap_upper-monthly.day_bootstrap_lower
    monthly.groupby(['task','model'])[['iid_width','block_width']].mean().to_csv(out/'mean_interval_widths.csv')
    plot_results(monthly,yearly,out)
    day_type_analysis(frames['demand'],args.day_types,out)
    metadata={'scope':'frozen models only; no adaptation or detector tuning',
              'fit_market_days':['2010-01-08','2013-12-31'],'validation_year':2014,
              'simulation':[str(start.date()),str(end.date())],'n_days':len(all_days),'retrain_count':0,
              'all_model_hashes_unchanged':True,'daily_official_checks':boundary_checks,
              'no_clipping':True,'bootstrap_replicates':args.bootstrap,'bootstrap_seed':9417,
              'bootstrap_unit':'whole market days, exactly 48 half-hours per day',
              'ci_interpretation':'expected monthly MAPE/accuracy under similar conditions, not prediction intervals; IID ignores dependence; daily bootstrap still assumes exchangeable independent days',
              'note':'2026 is Jan–Oct only. Results do not prove concept drift; no strategy chosen from this run.',
              'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__,
              'data_sha256':digest(args.data),'day_types_sha256':digest(args.day_types),
              'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'experiments/stage1_data.py',ROOT/'experiments/demand_models.py',ROOT/'experiments/demand_config.json',ROOT/'experiments/price_models.py']},
              'model_fitted_state_hashes':before,'warnings':caught,'elapsed_seconds':time.perf_counter()-begin}
    (out/'run_manifest.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
    print(summary.to_string(index=False),flush=True)
    print(f'Completed: {out}',flush=True)


if __name__=='__main__':
    main()
