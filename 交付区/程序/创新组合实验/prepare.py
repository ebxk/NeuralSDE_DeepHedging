"""Build and hash each fold's shared inputs, representation and market cache."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

from select_episodes import main as select
from contract_ledger import main as episode_ledger
from market_surfaces import main as surfaces


def digest(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""):
            h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf8")
    tmp.replace(path)


def identity(paths, config):
    return dict(config=config,files={str(Path(p).name):digest(p) for p in paths})


def reusable(marker, token, outputs):
    if not marker.exists() or any(not p.exists() for p in outputs):
        return False
    record=json.loads(marker.read_text(encoding="utf8"))
    return record["identity"]==token and record["outputs"]=={p.name:digest(p) for p in outputs}


def mark(marker,token,outputs):
    write_json(marker,dict(identity=token,outputs={p.name:digest(p) for p in outputs}))


def run(args):
    here = Path(__file__).resolve().parent
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    root = Path(args.out)
    root.mkdir(parents=True,exist_ok=True)
    revision = subprocess.check_output(["git","-C",args.upstream,"rev-parse","HEAD"],text=True).strip()
    assert revision=="6fa18790f14b56c99407ef987406459fa6e3b988",revision
    legacy = Path(args.legacy)
    source = identity([args.options,args.spot,*sorted(legacy.glob("*.py"))],dict(upstream=revision))
    surface = root/"surfaces.npz"
    marker = root/"surface.done.json"
    if not reusable(marker,source,[surface]):
        surfaces(args.options,args.spot,args.upstream,surface)
        mark(marker,source,[surface])
    failures = []
    for fold in cfg["folds"]:
        folder = root/str(fold[-1])
        folder.mkdir(exist_ok=True)
        fcfg = {**cfg,"split_ends":fold}
        config = folder/"config.json"
        write_json(config,fcfg)
        episodes, days = folder/"episodes.csv",folder/"episode_days.csv"
        marker = folder/"episodes.done.json"
        token = dict(source=source,fold=fold,windows=cfg["windows"],
                     moneyness=cfg["moneyness"],report=cfg["report_moneyness"])
        if not reusable(marker,token,[episodes,days]):
            select(args.options,args.spot,episodes,config)
            episode_ledger(args.options,args.spot,episodes,days)
            mark(marker,token,[episodes,days])
        if fold[-1]==2025 and args.old_data:
            assert digest(days)==digest(Path(args.old_data)/"episode_days.csv")
        reps = [folder/f"C{i}.npz" for i in range(3)]
        token = identity([here/"representation.py",surface,days],fcfg)
        marker = folder/"representation.done.json"
        if not reusable(marker,token,reps):
            subprocess.run([sys.executable,str(here/"representation.py"),
                str(surface),str(days),args.upstream,str(folder),str(config)],check=True)
            mark(marker,token,reps)
        for rep in reps:
            output = folder/rep.stem
            output.mkdir(exist_ok=True)
            token = identity([here/"market.py",rep,days,args.spot],dict(config=fcfg,upstream=revision))
            marker = output/"market.done.json"
            outputs = [output/n for n in ["common.npz","single.npz","market.json"]]
            if (output/"pool.npz").exists():
                outputs.append(output/"pool.npz")
            outputs.extend(sorted(output.glob("market_*.weights.h5")))
            if reusable(marker,token,outputs):
                continue
            marker.unlink(missing_ok=True)
            if (output/"pool.npz").exists():
                (output/"pool.npz").unlink()
            result = subprocess.run([sys.executable,str(here/"market.py"),str(rep),args.spot,
                 args.upstream,str(days),str(output),str(config)])
            if result.returncode:
                failures.append(f"{fold[-1]}/{rep.stem}")
                write_json(root/"market_failures.json",failures)
                continue
            outputs = [output/n for n in ["common.npz","single.npz","market.json"]]
            if (output/"pool.npz").exists():
                outputs.append(output/"pool.npz")
            outputs.extend(sorted(output.glob("market_*.weights.h5")))
            # Cache weights together with the exact generated paths.
            mark(marker,token,outputs)
        print("Prepared",fold,flush=True)
    write_json(root/"market_failures.json",failures)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("options","spot","upstream","legacy","config","out"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--old-data")
    run(p.parse_args())


