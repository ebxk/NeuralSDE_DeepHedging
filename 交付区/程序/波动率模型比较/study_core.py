"""Frozen identities and preregistered comparisons; no existence-only cache reuse."""
from hashlib import sha256
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = next(p for p in HERE.parents if (p/'AGENTS.md').exists() and (p/'工作区').is_dir())
OUT = PROJECT/'工作区/数据/处理后/波动率模型比较'
SHARED_SURFACES = OUT/'data/shared_surfaces.npz'

def base_factors(end):return OUT/f'representations/{end}/original.npz'
MODELS = ('gbm','nsde','original_linear','svi_linear','ssvi_linear',
          'essvi_linear','heston','localvol','sabr')
FOLDS = ((2022,2023,2025),(2019,2020,2021),(2020,2021,2022),(2021,2022,2023))
CONFIG = dict(study='volatility-20261112',folds=FOLDS,models=MODELS,
    seeds=[0,1,2],objectives=[0.,.1],inputs=['base','surface'],
    simulation_train=32768,simulation_valid=8192,epochs=100,updates_per_epoch=32,
    batch_size=1024,learning_rate=.003,validation_every=10,patience=3,
    width=46,cost_bp=5,holdout_fraction=.2,calibration_starts=5,
    calibration_seed=20261112,stale_days=5,substeps=2,diagnostic_substeps=4,
    bootstrap=2000,bootstrap_blocks=[1,3,6],financing_rate=0.,continuous_yield=0.,
    sabr_beta=1.,localvol_horizon_days=120,feature_version='shared-eight-v1')

def registry():
    return [dict(key=f'{end}_{m}_{info}_l{lam:g}_s{s}',fold=list(fold),end=end,
        model=m,information=info,lam=lam,seed=s,initial_key=f'{end}_s{s}')
        for fold in FOLDS for end in [fold[-1]] for m in MODELS
        for info in CONFIG['inputs'] for lam in CONFIG['objectives'] for s in CONFIG['seeds']]

def digest(path):
    h=sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''): h.update(block)
    return h.hexdigest()

def source_identity():
    return {str(p.relative_to(PROJECT)):digest(p) for p in sorted(HERE.glob('*.py'))}

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    tmp.replace(path)

def frozen_json(path,value):
    path=Path(path)
    canonical=json.loads(json.dumps(value))
    if path.exists():
        if json.loads(path.read_text(encoding='utf8'))!=canonical:
            raise ValueError(f'Frozen identity changed: {path}')
    else: write_json(path,canonical)

def quote_mask(dates,expiries,strikes):
    # Identical effective strikes/expiry/date have a single mask, independent of code/type.
    return np.array([int.from_bytes(sha256(f'{d}|{e}|{float(k):.3f}'.encode()).digest()[:8],
        'big')%5==0 for d,e,k in zip(dates,expiries,strikes)])

def bootstrap_legacy():
    import sys
    for name in ('完整认购','创新组合实验','扩展组合实验','瓶颈验证实验'):
        source=PROJECT/'工作区/当前研究/程序'/name
        if not source.exists():source=PROJECT/'交付区/程序'/name
        sys.path.insert(0,str(source))
    sys.path.insert(0,str(PROJECT/'工作区/运行支持/临时文件/neuralSDE-marketmodel'))

