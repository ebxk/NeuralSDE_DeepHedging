"""Stage completed, checked results locally and optionally synchronize safe aggregates."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from prepare_common import write_json

AGGREGATES = ("summary.csv","metrics_by_seed.csv","paired_intervals.csv","risk_tradeoff.png",
              "rolling_stability.png","comparison_5bp.png","diagnostics.json","completion.json","结果说明.md")
SOURCE = ("policy.py","train.py","market.py","prepare.py","prepare_common.py","representation.py",
          "report.py","dashboard.py","verify.py","deliver.py","study.json","run_study.ps1","运行说明.md")


def copy_files(files,destination):
    destination.mkdir(parents=True,exist_ok=True)
    for f in files:
        target=destination/f.name
        if f.resolve()!=target.resolve():
            shutil.copy2(f,target)


def git(repo,*args):
    return subprocess.check_output(["git","-C",str(repo),*args],encoding="utf8").strip()


def deliver(study,project,publish=False):
    here=Path(__file__).resolve().parent
    study,project=Path(study),Path(project)
    evaluation=study/"evaluation"
    complete=json.loads((evaluation/"completion.json").read_text(encoding="utf8"))
    assert complete["expected"]==1008 and complete["evaluated"]+complete["unavailable"]==1008 and not complete["failures"]
    sources=[here/n for n in SOURCE]
    aggregates=[evaluation/n for n in AGGREGATES]
    delivery=project/"交付区"
    program=delivery/"程序"/"创新组合实验"
    result=delivery/"结果"/"创新组合实验"
    weights=delivery/"参数与依赖"/"创新组合实验权重"
    copy_files(sources,program)
    copy_files(aggregates+[evaluation/n for n in ("completion.json","frozen_checkpoints.json","paths_index.json","failures.json")],result)
    for name in ("daily","paths"):
        shutil.copytree(evaluation/name,result/name,dirs_exist_ok=True)
    for folder in (d for d in study.iterdir() if d.name in {"2021","2022","2023","2025"}):
        copy_files(list(folder.glob("C?.npz"))+[folder/"config.json"],weights/folder.name)
        for rep in ("C0","C1","C2"):
            copy_files([*(folder/rep).glob("*.weights.h5"),folder/rep/"market.json",
                        folder/rep/"market.done.json"],weights/folder.name/rep)
        shutil.copytree(folder/"policies",weights/folder.name/"policies",dirs_exist_ok=True)
    copy_files([study/n for n in ("registry.json","legacy_replay.json","surface.done.json")],weights)
    status=dict(local=str(result),github=None)
    write_json(study/"delivery.json",status)
    if not publish:
        return
    expected="https://github.com/ebxk/NeuralSDE_DeepHedging.git"
    assert git(project,"remote","get-url","origin")==expected
    git(project,"fetch","origin","main")
    base=git(project,"rev-parse","FETCH_HEAD")
    with tempfile.TemporaryDirectory(prefix="nsde-study-") as temp:
        checkout=Path(temp)/"checkout"
        git(project,"worktree","add","--detach",str(checkout),base)
        try:
            public_program=checkout/program.relative_to(project)
            public_result=checkout/result.relative_to(project)
            copy_files(sources,public_program)
            copy_files(aggregates,public_result)
            # Keep the full breakdown locally. Tiny cells can reveal individual paths.
            for name in ("summary.csv","metrics_by_seed.csv"):
                path=public_result/name
                with path.open(encoding="utf8",newline="") as file:
                    reader=csv.DictReader(file)
                    fields=reader.fieldnames
                    rows=[r for r in reader if int(r["n"])>=10]
                with path.open("w",encoding="utf8",newline="") as file:
                    writer=csv.DictWriter(file,fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            public_paths=[(public_program/f.name).relative_to(checkout).as_posix() for f in sources]
            public_paths += [(public_result/f.name).relative_to(checkout).as_posix() for f in aggregates]
            git(checkout,"add","--",*public_paths)
            staged=git(checkout,"-c","core.quotePath=false","diff","--cached","--name-only").splitlines()
            assert set(staged).issubset(public_paths),staged
            if git(checkout,"status","--short"):
                git(checkout,"commit","-m","Deliver registered Neural SDE hedging combination study")
                git(checkout,"push","origin","HEAD:refs/heads/main")
            status["github"]=git(checkout,"rev-parse","HEAD")
            write_json(study/"delivery.json",status)
        finally:
            # Remove only the dedicated temporary worktree after git verifies it is clean.
            git(project,"worktree","remove",str(checkout))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("study")
    p.add_argument("project")
    p.add_argument("--publish",action="store_true")
    a=p.parse_args()
    deliver(a.study,a.project,a.publish)





