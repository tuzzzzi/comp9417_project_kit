"""Build the two inner ZIPs and Moodle's outer code/model ZIP.

No report is created or placed in any ZIP. Run after project_package/train.py.
"""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def make_zip(destination, files):
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path, name in sorted(files, key=lambda item: item[1]):
            archive.write(path, name)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f'Invalid ZIP: {destination}')


def main():
    out = ROOT / 'deliverables'
    out.mkdir(exist_ok=True)
    package = ROOT / 'project_package'
    required = ['model.py','train.py','features.py','demand.py','price.py','settings.json',
                'artifacts/demand.joblib','artifacts/price.joblib','artifacts/training.json']
    for name in required:
        if not (package/name).is_file():
            raise FileNotFoundError(f'Rebuild final artifacts first; missing {name}')
    artifacts_size = sum((package/name).stat().st_size for name in required if name.startswith('artifacts/'))
    if artifacts_size > 100*1024**2:
        raise ValueError('Final artifacts exceed 100 MB')
    submission = out / 'submission.zip'
    make_zip(submission, [(package/name,name) for name in required])

    code = []
    for directory in ['experiments','harness','template','example_package','project_package']:
        for path in (ROOT/directory).rglob('*'):
            rel=path.relative_to(ROOT)
            if path.is_file() and path.suffix in {'.py','.json'} and not any(
                part.startswith('.') or part in {'__pycache__','artifacts'} for part in rel.parts):
                code.append((path,str(rel)))
    for name in ['README.md','requirements.txt','FINAL_PROJECT_PLAN.md','MODEL_HANDOFF.md',
                 'starter_notebook.ipynb','eda_day_types.ipynb','day_types.csv','check_submission.py','data/README.md']:
        if (ROOT/name).is_file():code.append((ROOT/name,name))
    # A's fixed early models are small, required inputs to the frozen backtest.
    for path in (ROOT/'artifacts').glob('demand_*_frozen.joblib'):
        code.append((path,str(path.relative_to(ROOT))))
    experiment = out / 'experiment_code.zip'
    make_zip(experiment,code)
    outer = out / 'code_and_model.zip'
    make_zip(outer,[(experiment,experiment.name),(submission,submission.name)])
    with zipfile.ZipFile(outer) as archive:
        assert set(archive.namelist())=={'experiment_code.zip','submission.zip'}
    manifest={'moodle_upload_files':['code_and_model.zip','your group report PDF (supplied separately)'],
              'outer_zip_contents':['experiment_code.zip','submission.zip'],
              'report_in_zip':False,'final_artifacts_bytes':artifacts_size,
              'training_window':['2023-11-01','2026-10-31'],
              'sources':{'forum':'https://moodle.telt.unsw.edu.au/mod/forum/discuss.php?d=2821818',
                         'readme':'README.md sections 4–6'},
              'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [submission,experiment,outer]}}
    (out/'packaging.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for path in [submission,experiment,outer]:
        print(f'{path.name}: {path.stat().st_size/1024**2:.2f} MiB')


if __name__=='__main__':
    main()
