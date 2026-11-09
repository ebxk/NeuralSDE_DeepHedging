"""Local complete artifacts; only explicit sanitized aggregates may reach GitHub."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from extended_report import write_json


PUBLIC = ("summary.csv", "metrics_by_seed.csv", "paired_intervals.csv", "factorial_contrasts.csv",
          "mechanism_contrasts.csv", "legacy_comparison.csv", "completion.json", "report_manifest.json",
          "risk_tradeoff.png", "rolling_stability.png", "结果说明.md")
EXPECTED_REMOTE = "https://github.com/ebxk/NeuralSDE_DeepHedging.git"
PREVIOUS_DELIVERY = "b401d88f77e6d583bbe5f6fb18eba0a56b0ff82a"


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def sanitize_csv(source, destination):
    with source.open(encoding="utf8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if set(fields) & {"episode", "contract", "spot", "strike", "option_close", "pnl"}:
            raise ValueError(f"Detailed data cannot be published: {source.name}")
        rows = [r for r in reader if "n" not in r or (r["n"] and int(r["n"]) >= 2)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], encoding="utf8").strip()


def deliver(root, project, publish=False):
    root, project = Path(root).resolve(), Path(project).resolve()
    here = Path(__file__).resolve().parent
    evaluation = root / "evaluation"
    completion = json.loads((evaluation / "completion.json").read_text(encoding="utf8"))
    delivery = project / "交付区"
    program = delivery / "程序" / "扩展组合实验"
    result = delivery / "结果" / "扩展组合实验"
    weights = delivery / "参数与依赖" / "扩展组合实验"
    sources = sorted(f for f in here.iterdir() if f.is_file() and f.suffix in {".py", ".json", ".md", ".ps1"})
    for source in sources:
        copy(source, program / source.name)
    # Preserve prior studies and copy the detailed books only into the new local delivery.
    for file in evaluation.rglob("*"):
        if file.is_file() and "bootstrap_cache" not in file.relative_to(evaluation).parts:
            copy(file, result / file.relative_to(evaluation))
    for file in root.rglob("*"):
        if not file.is_file() or "evaluation" in file.relative_to(root).parts:
            continue
        if file.suffix == ".pt" or file.name.endswith(".weights.h5") or file.suffix == ".json":
            copy(file, weights / file.relative_to(root))
    status = dict(local=str(result), github=None, complete=completion["complete"])
    write_json(root / "delivery.json", status)
    if not publish:
        return status
    if not completion["complete"]:
        raise RuntimeError("Publish requires all registered tasks to be evaluated or explicitly unavailable")
    if git(project, "remote", "get-url", "origin") != EXPECTED_REMOTE:
        raise ValueError("Unexpected GitHub destination")
    git(project, "fetch", "origin", "main")
    base = git(project, "rev-parse", "FETCH_HEAD")
    subprocess.run(["git", "-C", str(project), "merge-base", "--is-ancestor", PREVIOUS_DELIVERY, base], check=True)
    temporary_root = project / "工作区" / "运行支持" / "临时文件"
    temporary_root.mkdir(parents=True, exist_ok=True)
    parent = Path(tempfile.mkdtemp(prefix="extended-public-", dir=temporary_root))
    checkout = parent / "checkout"
    git(project, "worktree", "add", "--detach", str(checkout), base)
    # An interrupted/failed publish retains its isolated checkout for diagnosis.
    public_program = checkout / program.relative_to(project)
    public_result = checkout / result.relative_to(project)
    allowed = []
    for source in sources:
        target = public_program / source.name
        copy(source, target)
        allowed.append(target.relative_to(checkout).as_posix())
    for name in PUBLIC:
        source, target = evaluation / name, public_result / name
        if not source.exists():
            raise FileNotFoundError(source)
        if source.suffix == ".csv":
            if name == "paired_intervals.csv":
                # The full predeclared comparison matrix can exceed GitHub's file limit.
                # Compress aggregates only after the same small-cell disclosure filter.
                safe = target.with_suffix(".safe.tmp")
                sanitize_csv(source, safe)
                target = target.with_suffix(".csv.gz")
                with safe.open("rb") as incoming, gzip.open(target, "wb", compresslevel=6) as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                safe.unlink()
            else:
                sanitize_csv(source, target)
        else:
            copy(source, target)
        allowed.append(target.relative_to(checkout).as_posix())
    git(checkout, "add", "--", *allowed)
    staged = git(checkout, "-c", "core.quotePath=false", "diff", "--cached", "--name-only").splitlines()
    if not set(staged).issubset(allowed):
        raise ValueError(f"Unexpected staged paths: {set(staged) - set(allowed)}")
    if staged:
        name = subprocess.run(["git", "-C", str(checkout), "config", "user.name"], capture_output=True, text=True)
        email = subprocess.run(["git", "-C", str(checkout), "config", "user.email"], capture_output=True, text=True)
        identity = [] if name.stdout.strip() and email.stdout.strip() else [
            "-c", "user.name=ebxk", "-c", "user.email=ebxk@users.noreply.github.com"]
        git(checkout, *identity, "commit", "-m", "Deliver extended Neural SDE hedging combination experiments")
        # Normal push is a fast-forward check; never force a concurrently changed remote.
        git(checkout, "push", "origin", "HEAD:refs/heads/main")
    status["github"] = git(checkout, "rev-parse", "HEAD")
    write_json(root / "delivery.json", status)
    git(project, "worktree", "remove", str(checkout))
    parent.rmdir()
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("project")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    print(json.dumps(deliver(args.root, args.project, args.publish), ensure_ascii=False))
