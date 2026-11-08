# Neural SDE × Deep Hedging 阶段研究快照

快照时间为 2026-11-09T01:32:45+08:00。完整实验仍在本地训练，本次发布用于审阅代码、实验设计和已验证的阶段结果。

## 当前进度与耗时

登记1,776项，已完成训练或核验复用 1358 项（76.5%），其余 418 项。已完成中包含 36 项旧检查点复用，快照未见模型失败记录。完整跨年回放和最终交付尚未生成。

| 评价时期 | 登记条目 | 已完成训练 |
|---|---:|---:|
| 2025 | 804 | 684 |
| 2021 | 324 | 324 |
| 2022 | 324 | 314 |
| 2023 | 324 | 36 |

当前为8路CPU并行，使用隔离的TensorFlow和PyTorch环境。已完成的新任务中位耗时约 16.8 分钟，包含真实验证选权重；每项最多100轮，每轮32次更新。教师监督、财富递归、历史网络会增加计算量。程序曾在11月8日凌晨约2:08停止，并于14:15从检查点恢复。停止原因尚未确认。此前对整套实验耗时的估计偏乐观。

## 已完成的阶段比较

2024–2025是已观察的历史评价集。本次阶段比较固定使用相同773条路径、5bp、MSE目标、三个种子。先保存144份检查点身份，再统一回放48组主配置。选权重仍使用2023年真实验证集，阶段结果没有改变后续训练参数。

| 配置 | RMSE | CVaR95损失 |
|---|---:|---:|
| 原C1覆盖加权宽网络 | 0.8769% | 1.3079% |
| 新增配置中RMSE最低的C2＋退火教师＋历史摘要 | 0.8964% | 1.5541% |
| C1＋曲面扩展＋教师＋财富＋历史完整组合 | 0.9474% | 1.8494% |

这48组中，其余47组的三个种子均值在两项指标上均未超过原C1宽网络。部分单种子有改善。新增配置中最低值是事后描述，不作为重新选模型的依据。尾部目标、GRU/残差及A/B连接对照、成本敏感性、完整跨年汇总仍待完成。全部阶段配置的配对区间见[阶段指标](阶段结果/summary_5bp_mse.csv)。

## 研究设计

C为三种因子权重。C0等权，C1合约覆盖，C2合约覆盖乘训练期对冲困难度。R扩展曲面，T退火BS教师，W自身交易前财富，H过去5/10日摘要。3×2⁴=48组主配置，在四个研究折、两个目标、三个种子下训练。原A为可靠性门控，B为KL受限稳健市场训练。

| 工作 | 登记训练条目 |
|---|---:|
| 主配置全矩阵 | 1,152 |
| 窄网络控制 | 144 |
| 教师预训练、常量监督和匹配真实风险控制 | 72 |
| 展平历史和GRU控制 | 48 |
| 同Delta输入与残差策略 | 48 |
| A/B连接和必要对照 | 168 |
| 0/10bp敏感性 | 144 |
| 合计 | 1,776 |

[任务清单](任务清单.csv)给出唯一作业键。[冻结配置](程序/study.json)给出训练预算、种子和四个时间折。每折完整路径均须在所属区间起止。主时期2015–2022训练2,626条，2023验证387条，2024–2025评价773条。滚动训练截止2019/2020/2021，对应验证2020/2021/2022、评价2021/2022/2023。

策略卖出认购，只交易ETF，持仓比例0–1，同收盘观察和成交，融资利率零。真实账本处理历史行权价与单位、登记资格、分红应收和到账、初末交易费用与经济到期支付。模拟连续市场使用无分红近似。

## 已实现及验证

- 原作者约束联合NSDE通过固定原源码适配。原作者版本为6fa18790f14b56c99407ef987406459fa6e3b988，实际ACT/365时间，两子步反射，保留配对四子步诊断。
- R扩展为35格点，期限7/14/30/60/90天，K/S=0.85至1.15、步长0.05。12组表示与参考市场、主折C1三成员池准备完成。四折训练期报价支持门槛均通过。
- 真实财富来自学生自身账户，包含分红应收，保持持仓及财富跨期梯度。门控专家读取组合策略共同实际持仓和财富，合成后一次收费。
- 教师标签仅训练期生成。主折87,303个到期前决策中79,421个标签有效。失效标签屏蔽监督，保留风险训练路径。常量、预训练及匹配真实风险控制均已实现。
- 主普通网络两层46单元，固定输入槽位。GRU16与同窗口展平网络匹配参数量。BS残差与同输入普通网络统一从重构曲面计算当期Delta。
- 修复原Local MV域外价格截平问题，价格、现货导数及因子导数使用同一外推分支。修复前结果保留，修复与扩网格分别比较。
- 9项策略/记账检查通过。旧18套模型经新接口和旧精度账本复现，最大误差约2.08×10⁻⁹。双精度现金账本与独立逐笔账本对齐。旧账本单精度与新双精度的纯舍入差另行记录。
- 已用387条真实验证路径跑通完整逐笔账本、汇总、图和本地展示页的端到端检查。
- 缓存身份包含来源、配置、父产物和专家哈希。恢复时清理旧辅助回放的当前引用，防止混入过期静态混合。

## 数值与实证限制

曲面坐标域外比例下降不等于对冲改善。R1全部12组的边界修正日比例高于R0；2023/C1配对两/四子步修正日比例为45.34%/54.79%，增量协方差亦有步长敏感性。这部分需与最终对冲结果一起审阅。

有限格点静态约束、连续模型近似、离散修正和动态无套利需要区分。同收盘成交是理想化条件。2024–2025已被反复观察，多项尝试的逐项区间不能解释为独立最终留出证据。种子不能视为新增市场样本。

## 文件入口

- [供GPT审阅的单文件材料](供GPT审阅.md)包含本说明及本轮完整源码和配置，可下载后上传到GPT网页端。
- [程序](程序)保留运行时源码的逐字快照。[源码哈希](源码哈希.json)可核对其与本地程序一致。
- [验证摘要](验证摘要.json)、[曲面报价支持度](曲面支持度.json)、[市场准备状态](市场准备状态.json)提供检查证据。
- 本轮程序依赖已发布的[创新组合程序](../../交付区/程序/创新组合实验)及[完整认购程序](../../交付区/程序/完整认购)。文件名有重名模块，运行时需保持原PYTHONPATH顺序。
- 此程序目录为审阅快照。恢复运行须放回项目工作区的扩展组合实验目录，并准备原本地输入和隔离环境。原始行情、逐路径损益、逐日行情账本和模型权重均留在本地。
- 上一次完整研究交付为b401d88，本次阶段发布保留该提交的全部内容。本地正在运行的程序、源配置和模型文件均未因同步而修改。

## 给GPT的审阅问题

请结合下列实际源码和阶段证据审查本研究，并区分已确认代码缺陷、统计设计问题和需要原始数据才能判断的事项。请重点检查训练/验证/历史评价隔离，曲面端点与外推导数，BS教师的训练标签与模拟/真实信息差异，自身财富和门控跨期梯度，KL权重及混合CVaR，固定输入掩码与模型容量，GRU窗口公平性，缓存恢复、冻结评价和多项比较。请解释新增组合目前没有超过强普通网络的可能原因，标明推测。请先检查已有实现再建议修改，并评价哪些进一步实验有识别价值。正在运行的登记实验继续执行，建议另列为后续方案。


## 本轮源码与配置全文

### 程序/expanded_market.py

````python
"""Prepare the R1 market artifacts without changing the frozen R0 files."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import traceback
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
UPSTREAM_REVISION = "6fa18790f14b56c99407ef987406459fa6e3b988"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def output_hashes(paths):
    from prepare_common import digest

    return {path.name: digest(path) for path in paths}


def reusable(marker, token):
    if not marker.is_file():
        return False
    try:
        record = read_json(marker)
        paths = [marker.parent / name for name in record["outputs"]]
        return record["identity"] == token and output_hashes(paths) == record["outputs"]
    except (KeyError, ValueError, OSError):
        return False


def mark(marker, token, paths, **status):
    from prepare_common import write_json

    write_json(marker, dict(identity=token, outputs=output_hashes(paths), **status))


def fit_market(factors, spot, upstream, days, out, config, pool):
    import market
    from prepare_common import write_json

    fit_member = market.fit_member

    class BootstrapNotRequested(ValueError):
        pass

    def single_member(*args):
        if args[-1]:
            raise BootstrapNotRequested()
        return fit_member(*args)

    if pool == "1":
        market.run(factors, spot, upstream, days, out, config)
    else:
        with patch.object(market, "fit_member", single_member):
            market.run(factors, spot, upstream, days, out, config)
    diagnostics_path = Path(out) / "market.json"
    diagnostics = read_json(diagnostics_path)
    if pool == "0":
        diagnostics["fit"] = [row for row in diagnostics["fit"] if row["seed"] == 2026]
    diagnostics["pool_requested"] = pool == "1"
    write_json(diagnostics_path, diagnostics)


def check_market(output, baseline, factor, diagnostics):
    import numpy as np

    with np.load(factor) as rep, np.load(output/"common.npz") as r1, np.load(baseline) as r0:
        for key in ("time", "warm", "strike", "premium", "real_dates", "return_dates", "returns"):
            np.testing.assert_array_equal(r1[key], r0[key], err_msg=key)
        np.testing.assert_array_equal(r1["real_dates"], rep["dates"])
        shape = r1["time"].shape
        for key in ("real_xi", "real_q", "gbm", "real_cov", "scale"):
            if not np.isfinite(r1[key]).all():
                raise ValueError(f"Nonfinite common array {key}")
    for name in ("single", "pool"):
        path = output/f"{name}.npz"
        if name == "pool" and diagnostics["members"] == 1:
            continue
        with np.load(path) as paths:
            if paths["spot"].shape != shape or paths["xi"].shape != (*shape, 2):
                raise ValueError(f"Unexpected simulated path shape for {name}")
            if not all(np.isfinite(paths[key]).all() for key in ("spot", "xi", "quality")):
                raise ValueError(f"Nonfinite simulated array in {name}")
    for row in diagnostics["numeric"]:
        if row["paired2"]["subdivisions"] != 2 or row["paired4"]["subdivisions"] != 4:
            raise ValueError("Missing paired two- and four-substep diagnostics")
        if min(row["full_slack"], row["fine_slack"]) < -1e-7:
            raise ValueError("Simulated factor path breaches static price constraints")


def prepare_market(factor, spot, upstream, days, output, config, baseline, token, pool, env):
    marker = output/"market.done.json"
    if reusable(marker, token):
        return read_json(marker)["status"]
    output.mkdir(exist_ok=True)
    marker.unlink(missing_ok=True)
    log = output/"market.log"
    with TemporaryDirectory(prefix="attempt-", dir=output) as temp, log.open("wb") as stream:
        staging = Path(temp)
        subprocess.run([sys.executable, "-u", str(HERE/"expanded_market.py"), "_fit",
                        str(factor), str(spot), str(upstream), str(days), str(staging),
                        str(config), str(int(pool))], stdout=stream, stderr=subprocess.STDOUT,
                       env=env, check=True)
        diagnostics = read_json(staging/"market.json")
        check_market(staging, baseline, factor, diagnostics)
        names = sorted(path.name for path in staging.iterdir())
        for name in names:
            (staging/name).replace(output/name)
        for path in [output/"pool.npz", *output.glob("market_*.weights.h5")]:
            if path.name not in names:
                path.unlink(missing_ok=True)
    pool_status = ("ready" if "pool.npz" in names else "unavailable") if pool else "not_requested"
    status = dict(single="ready", pool=pool_status, members=diagnostics["members"])
    mark(marker, token, [output/name for name in names], status=status)
    return status


def frozen_episodes(baseline, inputs, fold):
    days = baseline/"episode_days.csv"
    record = read_json(baseline/"episodes.done.json")
    baseline_inputs = record["identity"]["source"]["files"]
    if (record["identity"]["fold"] != fold
            or output_hashes([days])["episode_days.csv"] != record["outputs"]["episode_days.csv"]
            or any(baseline_inputs.get(name) != digest for name, digest in inputs.items())):
        raise ValueError(f"Frozen episode inputs do not match R0 for {fold}")
    return days


def prepare(args):
    project, out = args.project.resolve(), args.out.resolve()/"R1"
    work = project/"工作区"
    program = work/"当前研究"/"程序"
    original, legacy = program/"创新组合实验", program/"完整认购"
    upstream = work/"运行支持"/"临时文件"/"neuralSDE-marketmodel"
    baseline = work/"数据"/"处理后"/"创新组合实验"
    sys.path[:0] = [str(original), str(legacy)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, (original, legacy))),
           "PYTHONUTF8": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
           "MKL_NUM_THREADS": "2", "TF_NUM_INTEROP_THREADS": "2", "TF_CPP_MIN_LOG_LEVEL": "2"}
    from expanded_surfaces import SOURCE_FLAGS, fold_support, make_surfaces, read_quotes, source_report
    from prepare_common import write_json

    out.mkdir(parents=True, exist_ok=True)
    failure_file, status_file = out/"market_failures.json", out/"market_status.json"
    status = {}
    failures = []
    write_json(failure_file, failures)
    write_json(status_file, status)
    try:
        revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                                           text=True).strip()
        if revision != UPSTREAM_REVISION:
            raise ValueError(f"Unexpected upstream revision {revision}")
        cfg = read_json(original/"study.json")
        raw_files = output_hashes([args.options, args.spot])
        surface = out/"surfaces.npz"
        marker = out/"surface.done.json"
        token = dict(revision=revision, folds=cfg["folds"], inputs=raw_files,
                     code=output_hashes([HERE/"expanded_surfaces.py", legacy/"contract_ledger.py"]))
        support_path = out/"support.json"
        if not reusable(marker, token):
            marker.unlink(missing_ok=True)
            quotes = read_quotes(args.options, args.spot)
            support = {str(fold[-1]): fold_support(quotes, fold) for fold in cfg["folds"]}
            make_surfaces(quotes, upstream, surface)
            for fold in cfg["folds"]:
                support[str(fold[-1])]["source_by_split"] = source_report(surface, fold)
            write_json(support_path, dict(source_flags=SOURCE_FLAGS, folds=support))
            mark(marker, token, [surface, support_path])
        support = read_json(support_path)["folds"]
    except Exception:
        write_json(failure_file, [dict(stage="surface", error=traceback.format_exc())])
        raise

    def failure(stage, end, exc, rep=None):
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        failures.append(dict(stage=stage, fold=end, representation=rep, error=detail))
        write_json(failure_file, failures)
        print(f"R1 {end}/{rep or stage}: {exc}", flush=True)

    dependencies = output_hashes([HERE/"expanded_market.py", original/"market.py",
                                  original/"representation.py", legacy/"bs_baseline.py",
                                  legacy/"contract_ledger.py"])
    surface_hash = read_json(out/"surface.done.json")["outputs"]["surfaces.npz"]
    tasks = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        for fold in cfg["folds"]:
            end = str(fold[-1])
            folder = out/end
            folder.mkdir(exist_ok=True)
            try:
                if not support[end]["passed"]:
                    raise ValueError("Training-only support threshold failed. See R1/support.json")
                days = frozen_episodes(baseline/end, raw_files, fold)
                fcfg = {**cfg, "split_ends": fold}
                config = folder/"config.json"
                write_json(config, fcfg)
                reps = [folder/f"C{i}.npz" for i in range(3)]
                marker = folder/"representation.done.json"
                token = dict(config=fcfg, revision=revision, dependencies=dependencies,
                             surface=surface_hash, episodes=output_hashes([days]))
                if not reusable(marker, token):
                    marker.unlink(missing_ok=True)
                    subprocess.run([sys.executable, "-u", str(original/"representation.py"),
                                    str(surface), str(days), str(upstream), str(folder), str(config)],
                                   env=env, check=True)
                    mark(marker, token, reps)
            except Exception as exc:
                failure("representation", end, exc)
                continue
            print(f"R1 {end}: representations ready", flush=True)
            for factor in (reps[1], reps[0], reps[2]):
                rep = factor.stem
                pool = end == "2025" and rep == "C1"
                r0 = baseline/end/rep/"common.npz"
                identity = dict(representation=token, factor=output_hashes([factor]),
                                spot=raw_files[args.spot.name], pool=pool,
                                baseline=output_hashes([r0]))
                future = executor.submit(prepare_market, factor, args.spot, upstream, days,
                                         folder/rep, config, r0, identity, pool, env)
                tasks[future] = end, rep
        for future in as_completed(tasks):
            end, rep = tasks[future]
            try:
                status[f"{end}/{rep}"] = future.result()
                write_json(status_file, status)
                print(f"R1 {end}/{rep}: {status[f'{end}/{rep}']}", flush=True)
            except Exception as exc:
                failure("market", end, exc, rep)
    return not failures


if __name__ == "__main__":
    if sys.argv[1:2] == ["_fit"]:
        fit_market(*sys.argv[2:])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        for name in ("project", "out", "options", "spot"):
            parser.add_argument(f"--{name}", type=Path, required=True)
        sys.exit(0 if prepare(parser.parse_args()) else 1)

````

### 程序/expanded_surfaces.py

````python
"""Construct the fixed R1 price grid and audit its quote support."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from contract_ledger import terms


MATURITIES = np.array([7, 14, 30, 60, 90])
STRIKES = np.array([.85, .90, .95, 1., 1.05, 1.10, 1.15])
INTERPOLATED, ANCHOR, EXTRAPOLATED = 1, 2, 4
SOURCE_FLAGS = {
    "observed": 0, "interpolated": INTERPOLATED,
    "anchor": ANCHOR, "extrapolated": EXTRAPOLATED,
}


def read_quotes(option_file, spot_file):
    columns = ["date", "code", "trading_code", "list_date", "expire_date",
               "contract_type", "close", "volume", "position"]
    options = pd.read_csv(option_file, usecols=columns)
    spot = pd.read_csv(spot_file, encoding="gb18030").set_index("交易日期")["收盘价(元)"]
    options = options[(options.contract_type == "CO") & (options.close > 0)
                      & (options.volume > 0) & (options.position > 0)].copy()
    options["ttm"] = (pd.to_datetime(options.expire_date) -
                      pd.to_datetime(options.date)).dt.days
    options = options[options.ttm > 0].copy()
    options["spot"] = options.date.map(spot)
    if not np.isfinite(options.spot).all() or (options.spot <= 0).any():
        raise ValueError("Eligible option observations require a positive same-day spot close")
    options["strike"] = [float(terms(row, row["date"])[0]) for row in
                         options[["trading_code", "list_date", "expire_date", "date"]].to_dict("records")]
    options["k"] = options.strike / options.spot
    return options


def fold_support(quotes, fold):
    cutoff = f"{fold[0] + 1}-01-01"
    train = quotes[quotes.date < cutoff]
    days = train.date.nunique()
    if not days:
        raise ValueError(f"No eligible training quotes for {fold}")
    short = train.ttm.between(5, 9)
    short_days = int(train.loc[short, "date"].nunique())
    wing, nodes = [], []
    for k in STRIKES[[0, -1]]:
        near_k = (train.k - k).abs() <= .025 + 1e-12
        count = int(train.loc[near_k & train.ttm.between(1, 90), "date"].nunique())
        wing.append(dict(k=float(k), days=count, fraction=count/days, passed=count/days >= .50))
    for t in MATURITIES:
        for k in STRIKES:
            if t != 7 and k not in STRIKES[[0, -1]]:
                continue
            near = train[(train.ttm - t).abs().le(2) & (train.k - k).abs().le(.025 + 1e-12)]
            contracts = int(near.code.nunique())
            months = int(near.expire_date.str.slice(0, 7).nunique())
            nodes.append(dict(ttm=int(t), k=float(k), days=int(near.date.nunique()),
                              contracts=contracts, expiry_months=months,
                              passed=contracts >= 30 and months >= 6))
    tenor = dict(ttm=7, days=short_days, fraction=short_days/days,
                 passed=short_days/days >= .20)
    return dict(fold=fold, training_days=int(days), first_training_date=str(train.date.min()),
                last_training_date=str(train.date.max()), tenor=tenor, wings=wing, nodes=nodes,
                passed=tenor["passed"] and all(row["passed"] for row in wing + nodes),
                rules=dict(tenor_window_days=2, tenor_min_day_fraction=.20,
                           wing_moneyness_window=.025, wing_ttm_days=[1, 90],
                           wing_min_day_fraction=.50, node_min_contracts=30,
                           node_min_expiry_months=6, node_ttm_window_days=2,
                           node_moneyness_window=.025))


def quote_curve(quotes, spot, intrinsic):
    quotes = quotes.sort_values(["volume", "code"], ascending=[False, True])
    quotes = quotes.drop_duplicates("strike").sort_values("strike")
    k, c = quotes.k.to_numpy(), quotes.close.to_numpy()/spot
    slope = ((c[-1] - c[-2]) / (k[-1] - k[-2]) if len(k) > 1 else (c[-1] - 1) / k[-1])
    slope = np.clip(slope, -1, 0)
    curve = np.interp(STRIKES, np.r_[0., k], np.r_[1., c])
    right = STRIKES > k[-1]
    curve[right] = np.maximum(intrinsic[right], c[-1] + (STRIKES[right] - k[-1])*slope)
    exact = np.isclose(STRIKES[:, None], k, rtol=0, atol=1e-12).any(axis=1)
    source = np.full(len(STRIKES), INTERPOLATED, dtype=np.uint8)
    source[STRIKES < k[0]] = INTERPOLATED | ANCHOR | EXTRAPOLATED
    source[right] = EXTRAPOLATED
    source[exact] = 0
    return curve, source


def make_surfaces(quotes, upstream, output):
    sys.path.insert(0, str(upstream))
    from arbitragerepair.constraints import detect
    from arbitragerepair.repair import l1
    from cvxopt import solvers

    solvers.options["glpk"] = {"msg_lev": "GLP_MSG_OFF"}
    grid_t = np.repeat(MATURITIES/365, len(STRIKES))
    grid_k = np.tile(STRIKES, len(MATURITIES))
    intrinsic = np.maximum(1 - STRIKES, 0)
    days, spots, raw, repaired, sources = [], [], [], [], []
    for day, group in quotes.groupby("date", sort=True):
        spot = group.spot.iat[0]
        times, curves = [0], [intrinsic]
        provenance = [np.full(len(STRIKES), ANCHOR, dtype=np.uint8)]
        for ttm, quotes_at_t in group.groupby("ttm", sort=True):
            curve, source = quote_curve(quotes_at_t, spot, intrinsic)
            times.append(ttm)
            curves.append(curve)
            provenance.append(source)
        values = np.array([np.interp(MATURITIES, times, x) for x in np.array(curves).T]).T.ravel()
        flags = []
        for ttm in MATURITIES:
            hi = np.searchsorted(times, ttm)
            if hi == len(times):
                flags.append(provenance[-1] | EXTRAPOLATED)
            elif times[hi] == ttm:
                flags.append(provenance[hi])
            else:
                flags.append(provenance[hi - 1] | provenance[hi] | INTERPOLATED)
        if not days:
            A, b, counts, _ = detect(grid_t, grid_k, values)
        fixed = values + l1(A, b, values) if (A@values - b).min() < -1e-9 else values
        if not np.isfinite(fixed).all() or (A@fixed - b).min() < -1e-7:
            raise ValueError(f"Infeasible price repair on {day}")
        days.append(day)
        spots.append(spot)
        raw.append(values)
        repaired.append(fixed)
        sources.append(np.concatenate(flags))
    source = np.asarray(sources)
    np.savez_compressed(output, dates=np.array(days), spot=spots, raw=raw,
                        calls=repaired, source=source, extrapolated=(source & EXTRAPOLATED) != 0,
                        repaired=np.abs(np.asarray(repaired) - raw) > 1e-10,
                        A=A, b=b, constraint_counts=counts, grid_t=grid_t, grid_k=grid_k)
    print(f"{len(days)} surfaces, {len(grid_t)} nodes and {len(b)} constraints", flush=True)


def source_report(surface, fold):
    with np.load(surface) as data:
        years = data["dates"].astype("datetime64[Y]").astype(int) + 1970
        source = data["source"]
        repaired = data["repaired"]
        price_change = np.abs(data["calls"] - data["raw"])
        nodes = list(zip(data["grid_t"], data["grid_k"]))
    report = {}
    previous = 0
    for phase, end in zip(("train", "validation", "test"), fold):
        rows = (years > previous) & (years <= end)
        previous = end
        report[phase] = [
            dict(ttm=round(t*365), k=k, days=int(rows.sum()),
                 observed=int((source[rows, j] == 0).sum()),
                 interpolated=int(((source[rows, j] & INTERPOLATED) != 0).sum()),
                 anchor=int(((source[rows, j] & ANCHOR) != 0).sum()),
                 extrapolated=int(((source[rows, j] & EXTRAPOLATED) != 0).sum()),
                 repaired=int(repaired[rows, j].sum()),
                 mean_abs_repair=float(price_change[rows, j].mean()))
            for j, (t, k) in enumerate(nodes)
        ]
    return report

````

### 程序/extended_curve.py

````python
"""Shared constrained curve evaluation for real and simulated states."""
import numpy as np
import torch


def curve(tau, money, xi, rep, scale):
    """Return normalized call, strike slope and normalized-factor derivatives."""
    times = np.unique(rep["grid_t"])
    strikes = np.unique(rep["grid_k"])
    shape = tau.shape
    tau, money, xi = tau.ravel(), money.ravel(), xi.reshape(-1, 2)
    centers = rep["center"].reshape(len(times), len(strikes))
    basis = (rep["basis"] * scale[:, None]).reshape(2, len(times), len(strikes))
    n = len(tau)
    t1 = np.minimum(np.searchsorted(times, tau), len(times)-1)
    t0 = t1-1
    a = np.clip((tau-np.r_[0, times][t1])/(times[t1]-np.r_[0, times][t1]), 0, 1)
    j = np.clip(np.searchsorted(strikes, money, side="right")-1, 0, len(strikes)-2)
    k0, k1 = strikes[j], strikes[j+1]
    payoff = np.maximum(1-money, 0)
    zero = np.zeros((n, 2))

    def at(t):
        idx = np.maximum(t, 0)
        p = centers[idx, j] + np.sum(xi*basis[:, idx, j].T, axis=1)
        b = basis[:, idx, j].T
        d = basis[:, idx, j+1].T-b
        q = centers[idx, j+1] - centers[idx, j] + np.sum(xi*d, axis=1)
        slope = q/(k1-k0)
        d = d/(k1-k0)[:, None]
        price = p+slope*(money-k0)
        deriv = b+d*(money-k0)[:, None]
        left = money < strikes[0]
        price[left] = 1+(p[left]-1)*money[left]/strikes[0]
        slope[left] = (p[left]-1)/strikes[0]
        deriv[left] = b[left]*money[left, None]/strikes[0]
        right = money > strikes[-1]
        constrained = right & ((slope < -1) | (slope > 0))
        price[constrained] = (p+q)[constrained] + np.clip(slope[constrained], -1, 0)*(money[constrained]-k1[constrained])
        deriv[constrained] = (b+d*(k1-k0)[:, None])[constrained]
        slope[constrained] = np.clip(slope[constrained], -1, 0)
        floor = price < payoff
        cap = price > 1
        intrinsic = floor | (t < 0)
        price[intrinsic], slope[intrinsic], deriv[intrinsic] = payoff[intrinsic], -(money[intrinsic] < 1).astype(float), 0
        cap &= ~intrinsic
        price[cap], slope[cap], deriv[cap] = 1, 0, 0
        return price, slope, deriv

    p0, s0, d0 = at(t0)
    p1, s1, d1 = at(t1)
    return ((p0+(p1-p0)*a).reshape(shape), (s0+(s1-s0)*a).reshape(shape),
            (d0+(d1-d0)*a[:, None]).reshape(*shape, 2))


def bs_delta(spot, strike, quote, tau):
    """Zero-rate, zero-yield implied BS delta; bad quotes carry a false mask."""
    spot, strike, quote, tau = (torch.as_tensor(x, dtype=torch.float64)
                                for x in (spot, strike, quote, tau))
    k, c = strike/spot, quote/spot
    valid = (tau > 0) & (k > 0) & (quote > (spot-strike).clamp(min=0)) & (quote < spot) & torch.isfinite(c)
    result = torch.zeros_like(c, dtype=torch.float32)
    if valid.any():
        m, c0 = -k[valid].log(), c[valid]
        k0 = k[valid]
        lo, hi = torch.zeros_like(c0), torch.full_like(c0, 12.)
        for _ in range(32):
            vol = (lo+hi)/2
            d1 = m/vol+vol/2
            call = torch.special.ndtr(d1)-k0*torch.special.ndtr(d1-vol)
            lo = torch.where(call < c0, vol, lo)
            hi = torch.where(call < c0, hi, vol)
        vol = (lo+hi)/2
        d1 = m/vol+vol/2
        d = torch.special.ndtr(d1)
        good = torch.isfinite(d) & ((d-k0*torch.special.ndtr(d1-vol)-c0).abs() <= 1e-7)
        result[valid] = d.to(torch.float32)
        valid[valid.clone()] = good
    return result, valid


def reconstructed_delta(spot, strike, tau, xi, rep, scale):
    outputs, masks = [], []
    strike = np.broadcast_to(strike, spot.shape)
    for sl in np.array_split(np.arange(len(spot)), max(1, (len(spot)+511)//512)):
        s, k, t = spot[sl], strike[sl], tau[sl]
        c, _, _ = curve(t, k/s, xi[sl], rep, scale)
        value, ok = bs_delta(s, k, c*s, t)
        # Current intrinsic delta is the finite fallback; same in both branches.
        outputs.append(torch.where(ok, value, torch.from_numpy((s > k).astype("f"))))
        masks.append(ok)
    return torch.cat(outputs), torch.cat(masks)


````

### 程序/extended_data.py

````python
"""Read frozen inputs and construct causal path tensors for the expanded study."""
from pathlib import Path

import numpy as np
import torch

from contract_ledger import EVENTS, ledger
from hedge_experiment import features, read_paths
from train import make_real
from extended_curve import bs_delta, reconstructed_delta


def load_npz(path):
    with np.load(path, allow_pickle=False) as file:
        return dict(file)


def historical(common):
    days = common["return_dates"]
    ret = common["returns"]
    factors = np.full((len(days), 2), np.nan)
    ix = np.searchsorted(days, common["real_dates"])
    assert np.array_equal(days[ix], common["real_dates"])
    factors[ix] = common["real_xi"]
    rv = np.full(len(days), np.nan)
    rv[19:] = np.sqrt(252*np.mean(np.lib.stride_tricks.sliding_window_view(ret, 20)**2, axis=-1))
    tape = np.column_stack([20*ret, np.diff(factors, axis=0, prepend=np.nan),
                            4*np.diff(rv, prepend=np.nan)]).astype("f")
    return days, tape


def aligned_history(index, tape):
    grid = index[..., None] + np.arange(-9, 1)
    out = tape[grid.clip(0)]
    return np.where((grid >= 0)[..., None], out, np.nan)


def batch(data, rows, history=False):
    result = {key: value[rows] for key, value in data.items() if key != "tape"}
    if history:
        windows = data["tape"][rows].unfold(1, 10, 1).transpose(-1, -2)
        good = windows.isfinite()
        result["history"] = torch.where(good, windows, 0)
        result["history_valid"] = good
    return result


class Inputs:
    def __init__(self, episode_root, directory, rep, fold):
        self.episode_file = episode_root/"episode_days.csv"
        self.directory = directory
        self.rep = load_npz(rep)
        self.common = load_npz(directory/"common.npz")
        self.folds = fold
        self.paths = dict(zip(fold, read_paths(self.episode_file, fold)))
        self.calendar, self.tape = historical(self.common)
        self.real = {}
        self._sim = {}

    def simulation(self, market, need_delta=False):
        name = "single" if market == "single" else "pool"
        if name not in self._sim:
            d = load_npz(self.directory/f"{name}.npz")
            s, xi = d["spot"], d["xi"]
            t, warm = self.common["time"], self.common["warm"]
            strike = self.common["strike"].astype("f")
            tau = t[:, -1:] - t
            returns = np.diff(np.log(s), axis=1)
            rv = np.sqrt(252*np.mean(np.lib.stride_tricks.sliding_window_view(
                np.concatenate([warm, returns], axis=1), 20, axis=1)**2, axis=-1))
            active = np.diff(t, axis=1) > 0
            f = features(s, self.common["strike"][:, None], tau, rv,
                         np.concatenate([xi, d["quality"]], axis=-1))[:, :-1].copy()
            origins = np.random.default_rng(104).integers(len(self.paths[self.folds[0]]), size=len(s))
            start_dates = np.array([self.paths[self.folds[0]][i][0]["date"] for i in origins])
            ix = np.searchsorted(self.calendar, start_dates)
            assert np.array_equal(self.calendar[ix], start_dates)
            # The unchanged simulator samples episode groups in integer ID order.
            assert np.allclose(s[:, 0], [float(self.paths[self.folds[0]][i][0]["spot"]) for i in origins])
            assert np.allclose(strike, [float(self.paths[self.folds[0]][i][0]["strike"]) for i in origins])
            warm_history = aligned_history(ix, self.tape)
            future = np.concatenate([20*returns[..., None], np.diff(xi, axis=1),
                                     4*np.diff(rv, axis=1)[..., None]], axis=2)
            future = np.where(active[..., None], future, np.nan)
            tape = np.concatenate([warm_history, future[:, :-1]], axis=1).astype("f")
            self._sim[name] = (dict(f=torch.from_numpy(f), spot=torch.from_numpy(s),
                unit=torch.ones_like(torch.from_numpy(s)), active=torch.from_numpy(active),
                premium=torch.from_numpy(self.common["premium"].astype("f")),
                strike_end=torch.from_numpy(strike), tape=torch.from_numpy(tape)),
                d["member"], xi, tau)
        data, ids, xi, tau = self._sim[name]
        if need_delta and "delta" not in data:
            data["delta"], data["delta_valid"] = reconstructed_delta(
                data["spot"].numpy()[:, :-1], self.common["strike"][:, None],
                tau[:, :-1], xi[:, :-1], self.rep, self.common["scale"])
        return data, ids

    def observed(self, year, need_delta=False, teacher=False):
        if teacher and year != self.folds[0]:
            raise ValueError("Teacher labels can only use the training fold")
        paths = self.paths[year]
        if year not in self.real:
            f, _ = make_real(paths, self.common, self.episode_file)
            n, h, _ = f.shape
            shape = (n, h+1)
            s, u, k, quote = (np.empty(shape, dtype="d") for _ in range(4))
            tau = np.zeros(shape)
            active = np.zeros((n, h), dtype=bool)
            tape = np.full((n, h+9, 4), np.nan, dtype="f")
            event = {key: np.zeros(shape, dtype="d" if key == "record_amount" else "int64")
                     for key in ("record_amount", "record_event", "ex_event", "pay_event")}
            for i, path in enumerate(paths):
                size = len(path)
                days = np.array([row["date"] for row in path])
                ix = np.searchsorted(self.calendar, days)
                assert np.array_equal(self.calendar[ix], days)
                for arr, key in [(s, "spot"), (u, "unit"), (k, "strike"), (quote, "option_close")]:
                    arr[i, :size] = [float(row[key]) for row in path]
                    arr[i, size:] = arr[i, size-1]
                tau[i, :size] = (days[-1].astype("datetime64[D]")-days.astype("datetime64[D]")).astype(float)/365
                active[i, :size-1] = True
                tape[i, :10] = aligned_history(ix[0], self.tape)
                tape[i, 10:size+8] = self.tape[ix[1:-1]]
                for j, (ex, record, pay, amount, _) in enumerate(EVENTS, 1):
                    for name, date in (("record_event", record), ("ex_event", ex), ("pay_event", pay)):
                        event[name][i, :size][days == date] = j
                    event["record_amount"][i, :size][days == record] = float(amount)
            data = dict(f=f, spot=torch.from_numpy(s), unit=torch.from_numpy(u),
                premium=torch.from_numpy(quote[:, 0].copy()), strike_end=torch.from_numpy(k[:, -1].copy()),
                active=torch.from_numpy(active), tape=torch.from_numpy(tape),
                **{key: torch.from_numpy(val) for key, val in event.items()})
            self.real[year] = data, k, tau, quote
        data, k, tau, quote = self.real[year]
        if need_delta and "delta" not in data:
            data["delta"], data["delta_valid"] = reconstructed_delta(
                data["spot"].numpy()[:, :-1], k[:, :-1], tau[:, :-1],
                data["f"].numpy()[:, :, 3:5], self.rep, self.common["scale"])
        if teacher and "teacher" not in data:
            data["teacher"], data["teacher_valid"] = bs_delta(
                data["spot"][:, :-1], k[:, :-1], quote[:, :-1], tau[:, :-1])
            data["teacher_valid"] &= data["active"]
        return paths, data


def books(paths, ratios, gate, cost):
    result = []
    for i, path in enumerate(paths):
        entries = ledger(path, map(float, ratios[i, :len(path)-1]), float(path[0]["option_close"]), cost/10000)
        for j, row in enumerate(entries):
            row["gate_weight"] = float(gate[i, j]) if gate is not None and j < len(path)-1 else ""
        result.append(entries)
    return result


````

### 程序/extended_delivery.py

````python
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

````

### 程序/extended_policy.py

````python
"""Extended policies and a differentiable, self-financing contract ledger.

Historical windows contain four causal increments: return, two factor changes,
and realized-volatility change. Account wealth is scaled by 100 only at the
network boundary; all cash flows retain their original monetary units.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F


def _count(inputs, width, single=False):
    return (inputs+1)*width + width+1 + (0 if single else (width+1)*width)


def _width(inputs, target, extra=0):
    width = min(range(4, 257), key=lambda w: abs(_count(inputs, w)+extra-target))
    if abs(_count(inputs, width)+extra-target)/target > .05:
        raise ValueError("No integer hidden width matches the parameter budget")
    return width


def _history(data, dtype, base=False):
    values = data["history"].to(dtype=dtype)
    valid = data["history_valid"].bool()
    if valid.ndim == values.ndim-1:
        valid = valid[..., None].expand_as(values)
    if base:
        valid = valid.clone()
        valid[..., 1:3] = False
    return torch.where(valid, values, torch.zeros_like(values)), valid


def _summaries(values, valid):
    output = []
    for length in (5, 10):
        window, mask = values[..., -length:, :], valid[..., -length:, :]
        output.append(window.sum(dim=-2))
        variance = window[..., 0].square().sum(dim=-1)/mask[..., 0].sum(dim=-1).clamp_min(1)
        output += [variance.sqrt()[..., None], mask.to(values.dtype).mean(dim=(-1, -2))[..., None]]
    return torch.cat(output, dim=-1)


class DensePolicy(nn.Module):
    def __init__(self, kind, wealth=False, history=False, width=46,
                 current=8, single=False, target=None):
        super().__init__()
        self.kind, self.wealth, self.history = kind, wealth, history or kind in {"flat", "gru"}
        self.current = current
        extra, count = 0, current
        if kind == "gru":
            self.encoder = nn.GRU(8, 16, batch_first=True)
            extra = sum(p.numel() for p in self.encoder.parameters())
            count += 16
        elif kind == "flat":
            count += 80
        else:
            count += 12
        if kind in {"delta_input", "delta_residual"}:
            count += 2
        self.exogenous_size = count
        # Wealth/history slots stay fixed across their on/off ablations.
        inputs = count + 2
        if target is not None:
            width = _width(inputs, target, extra)
        layers = [nn.Linear(inputs, width), nn.Tanh()]
        if not single:
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1), nn.Sigmoid()]
        self.network = nn.Sequential(*layers)
        if single or kind == "delta_residual":
            nn.init.zeros_(self.network[-2].weight)
            nn.init.zeros_(self.network[-2].bias)

    def prepare(self, data):
        dtype = self.network[0].weight.dtype
        pieces = [data["f"][..., :self.current].to(dtype=dtype)]
        columns = list(range(self.current))
        if self.history:
            values, valid = _history(data, dtype, base=self.kind == "base")
            if self.kind in {"flat", "gru"}:
                frames = torch.cat([values, valid.to(dtype)], dim=-1)
                if self.kind == "flat":
                    pieces.append(frames.flatten(start_dim=-2))
                else:
                    shape = frames.shape[:2]
                    _, hidden = self.encoder(frames.reshape(-1, 10, 8))
                    pieces.append(hidden[0].reshape(*shape, 16))
                columns += list(range(self.current, self.current + (80 if self.kind == "flat" else 16)))
            else:
                pieces.append(_summaries(values, valid))
                columns += list(range(self.current, self.current+12))
        if self.kind in {"delta_input", "delta_residual"}:
            delta = data["delta"].to(dtype=dtype)
            pieces += [delta[..., None], data["delta_valid"].to(dtype)[..., None]]
            columns += [self.exogenous_size-2, self.exogenous_size-1]
        features = torch.cat(pieces, dim=-1).transpose(0, 1).contiguous()
        weights = self.network[0].weight
        weights = weights[:, :len(columns)] if columns == list(range(len(columns))) else weights[:, columns]
        result = F.linear(features, weights, self.network[0].bias)
        if self.kind == "delta_residual":
            return result, torch.logit(delta.clamp(1e-4, 1-1e-4)).transpose(0, 1)
        return result

    def action(self, prepared, previous, wealth):
        offset = None
        if self.kind == "delta_residual":
            prepared, offset = prepared
        dtype = self.network[0].weight.dtype
        h = prepared + previous.to(dtype)[:, None]*self.network[0].weight[:, -1]
        if self.wealth:
            h = h + (100*wealth.to(dtype))[:, None]*self.network[0].weight[:, -2]
        for layer in list(self.network)[1:-1]:
            h = layer(h)
        logits = h.squeeze(-1)
        return torch.sigmoid(logits if offset is None else logits+offset), None


class BlendPolicy(nn.Module):
    def __init__(self, kind, wealth, history, experts, fixed):
        super().__init__()
        if experts is None or "base" not in experts or not ({"factor", "wide"} & set(experts)):
            raise ValueError("A blend requires base and factor (or wide) experts")
        self.kind, self.fixed, self.wealth = kind, fixed, wealth
        self.base = copy.deepcopy(experts["base"]).requires_grad_(False)
        self.factor = copy.deepcopy(experts.get("factor", experts.get("wide"))).requires_grad_(False)
        self.gate = None if fixed is not None else DensePolicy(
            "gate", wealth, history, width=8, current=5 if kind == "state_gate" else 8, single=True)

    def prepare(self, data):
        return self.base.prepare(data), self.factor.prepare(data), self.gate.prepare(data) if self.gate else None

    def action(self, prepared, previous, wealth):
        base, factor, gate = prepared
        a, _ = self.base.action(base, previous, wealth)
        b, _ = self.factor.action(factor, previous, wealth)
        weight = self.gate.action(gate, previous, wealth)[0] if self.gate else torch.full_like(a, self.fixed)
        return torch.lerp(a, b, weight), weight


def make_policy(kind, wealth=False, history=False, experts=None, fixed=None):
    if kind in {"state_gate", "reliable_gate", "static"}:
        if kind == "static" and fixed is None:
            fixed = .5
        if fixed is not None and not 0 <= fixed <= 1:
            raise ValueError("A static expert weight must lie in [0,1]")
        return BlendPolicy(kind, wealth, history, experts, fixed)
    if kind not in {"wide", "narrow", "base", "capacity", "flat", "gru", "delta_input", "delta_residual"}:
        raise ValueError(f"Unknown policy {kind}")
    target = None
    if kind in {"flat", "gru"}:
        target = _count(8+12+2, 46)
    elif kind == "capacity":
        if experts is None:
            experts = dict(base=make_policy("base", wealth, history), factor=make_policy("wide", wealth, history))
        target = sum(p.numel() for m in experts.values() for p in m.parameters())
        target += _count(8+12+2, 8, single=True)
    return DensePolicy(kind, wealth, history, width=32 if kind in {"base", "narrow"} else 46,
                       current=3 if kind == "base" else 8, target=target)


def load_legacy(model, state):
    """Map original Sequential keys without changing the original checkpoint."""
    if not isinstance(model, DensePolicy) or model.wealth or model.history:
        raise ValueError("Legacy checkpoints require a plain current-state policy")
    mapped = dict(state)
    original = state["0.weight"]
    target = torch.zeros_like(model.network[0].weight)
    if original.shape[1] != model.current+1 or original.shape[0] != target.shape[0]:
        raise ValueError("Legacy input count or hidden width does not match")
    target[:, :model.current] = original[:, :-1]
    target[:, -1] = original[:, -1]
    mapped["0.weight"] = target
    model.network.load_state_dict(mapped)
    return model


def _at(prepared, index):
    if isinstance(prepared, tuple):
        return tuple(_at(value, index) for value in prepared)
    return None if prepared is None else prepared[index]


def simulated_pnl(model, data, fee):
    """Equivalent normalized wealth recursion for unit-one, no-dividend simulations."""
    spot, active = data["spot"], data["active"]
    price = spot/spot[:, :1]
    premium = data["premium"]/spot[:, 0]
    payoff = torch.relu(spot[:, -1]-data["strike_end"])/spot[:, 0]
    prepared = model.prepare(data)
    previous = torch.zeros_like(premium)
    if model.wealth:
        wealth = premium
    else:
        wealth = premium-payoff
    for t in range(active.shape[1]):
        account = wealth if model.wealth else None
        target, _ = model.action(_at(prepared, t), previous, account)
        target = torch.where(active[:, t], target, previous)
        wealth = wealth + target*(price[:, t+1]-price[:, t]) - fee*(target-previous).abs()*price[:, t]
        previous = target
    return wealth - fee*previous*price[:, -1] - (payoff if model.wealth else 0.)


def rollout(model, data, fee, capture=False):
    """Return normalized terminal wealth, active hedge ratios and gate weights.

    Floating point ledger precision follows ``spot`` (float64 for real paths).
    Event IDs are positive integers, with zero indicating no event. Each row's
    active decisions form a prefix; its first inactive day is settlement.
    """
    spot, unit, active = data["spot"], data["unit"], data["active"].bool()
    n, horizon = active.shape
    dtype, device = spot.dtype, spot.device
    if unit.shape != spot.shape or spot.shape != (n, horizon+1):
        raise ValueError("Spot and current contract units must include the terminal day")
    if horizon > 1 and (active[:, 1:] & ~active[:, :-1]).any():
        raise ValueError("Active decisions must form a prefix")
    prepared = model.prepare(data)
    initial_notional = spot[:, 0]*unit[:, 0]
    cash = data["premium"].to(dtype=dtype)*unit[:, 0]
    shares = torch.zeros(n, dtype=dtype, device=device)
    receivable = torch.zeros_like(shares)
    settled = torch.zeros(n, dtype=torch.bool, device=device)
    zeros = torch.zeros_like(active[:, 0]) if horizon else settled
    record, ex, pay = (data.get(name) for name in ("record_event", "ex_event", "pay_event"))
    event_ids = sorted(set(record[record > 0].detach().cpu().tolist())) if record is not None else []
    entitlements = {event: torch.zeros_like(shares) for event in event_ids}
    ratios, weights = [], []
    for t in range(horizon+1):
        if event_ids:
            accrued = sum(torch.where(ex[:, t] == event, amount, torch.zeros_like(amount))
                          for event, amount in entitlements.items()) if ex is not None else 0
            paid = sum(torch.where(pay[:, t] == event, amount, torch.zeros_like(amount))
                       for event, amount in entitlements.items()) if pay is not None else 0
            receivable = receivable + accrued-paid
            cash = cash + paid
        decision = active[:, t] if t < horizon else zeros
        settlement = ~decision & ~settled
        if t < horizon:
            model_dtype = next(model.parameters()).dtype
            previous = shares.to(model_dtype)/unit[:, t].to(model_dtype)
            wealth = (cash+receivable+shares*spot[:, t])/initial_notional
            ratio, weight = model.action(_at(prepared, t), previous, wealth)
            ratio = torch.where(decision, ratio, torch.zeros_like(ratio))
            ratios.append(ratio)
            if capture and weight is not None:
                weights.append(torch.where(decision, weight, torch.zeros_like(weight)))
            target = ratio.to(dtype)*unit[:, t]
        else:
            target = torch.zeros_like(shares)
        traded = target-shares
        cash = cash-traded*spot[:, t]-fee*traded.abs()*spot[:, t]
        payoff = unit[:, t]*torch.relu(spot[:, t]-data["strike_end"].to(dtype=dtype))
        cash = cash-torch.where(settlement, payoff, torch.zeros_like(payoff))
        shares = target
        for event in event_ids:
            entitlements[event] = torch.where(record[:, t] == event,
                shares*data["record_amount"][:, t].to(dtype=dtype), entitlements[event])
        settled = settled | settlement
    pnl = (cash+receivable+shares*spot[:, -1])/initial_notional
    ratio_tensor = torch.stack(ratios, dim=1) if ratios else spot[:, :0]
    return pnl, ratio_tensor, torch.stack(weights, dim=1) if weights else None

````

### 程序/extended_registry.py

````python
"""Fixed model registry; one key per distinct training job."""
from dataclasses import asdict, dataclass
from itertools import product


@dataclass(frozen=True)
class Job:
    end: int
    rep: str
    r: int
    t: str
    w: bool
    h: bool
    cost: int
    lam: float
    seed: int
    market: str
    policy: str
    family: str

    @property
    def key(self):
        return (f"{self.end}_R{self.r}_{self.rep}_{self.t}_W{int(self.w)}H{int(self.h)}_"
                f"{self.cost}bp_L{self.lam:g}_S{self.seed}_{self.market}_{self.policy}")

    def meta(self):
        return asdict(self)


def registry(cfg):
    result = {}

    def add(family, end, rep, r, t, w, h, cost=5, market="single", policy="wide"):
        for lam, seed in product(cfg["lambdas"], cfg["seeds"]):
            j = Job(end, rep, r, t, w, h, cost, lam, seed, market, policy, family)
            result.setdefault(j.key, j)

    reps = ("C0", "C1", "C2")
    bits = (False, True)
    for _, _, end in cfg["folds"]:
        for rep, r, t, w, h in product(reps, (0, 1), ("off", "anneal"), bits, bits):
            add("main", end, rep, r, t, w, h)
        for rep, r in product(reps, (0, 1)):
            add("narrow", end, rep, r, "off", False, False, policy="narrow")

    for r, enabled in product((0, 1), bits):
        for t in ("warm", "constant", "real"):
            add("teacher", 2025, "C1", r, t, enabled, enabled)
        for policy in ("flat", "gru"):
            add("history", 2025, "C1", r, "anneal" if enabled else "off", True, True, policy=policy)
        for policy in ("delta_input", "delta_residual"):
            add("delta", 2025, "C1", r, "anneal" if enabled else "off", enabled, enabled, policy=policy)
    for r, market in product((0, 1), ("single", "equal", "robust")):
        for policy in ("base", "wide", "state_gate", "reliable_gate", "capacity"):
            add("bridge", 2025, "C1", r, "anneal" if r else "off", bool(r), bool(r),
                market=market, policy=policy)
    for rep, r, enabled, cost in product(reps, (0, 1), bits, (0, 10)):
        add("cost", 2025, rep, r, "anneal" if enabled else "off", enabled, enabled, cost=cost)
    assert len(result) == 1776
    return result


````

### 程序/extended_replay.py

````python
"""Frozen policy replay and corrected model sensitivities."""
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import numpy as np

from prepare_common import digest, write_json
from report import baseline_book, bs_path, export
from train import score
from extended_curve import curve
from extended_data import books
from extended_train import DELTA, GATES, expert_jobs, predictions, restore, state_file


def folders(out):
    for name in ("paths", "daily"):
        (out/name).mkdir(parents=True, exist_ok=True)


def replay(root, out, inputs, job, record, fixed=None, tag=None):
    model = restore(root, job, fixed)
    paths, data = inputs.observed(job.end, job.policy in DELTA)
    key = job.key if tag is None else f"{job.key}_{tag}"
    dest = out/"paths"/f"{key}.json"
    daily = out/"daily"/f"{key}.csv.gz"
    mark = dict(weights=record.get("weights_sha256"), identity=record.get("identity"), fixed=fixed,
                replay_source=digest(__file__), export_source=digest(Path(export.__code__.co_filename)))
    if dest.exists() and daily.exists():
        cached = json.loads(dest.read_text(encoding="utf8"))
        if cached["meta"].get("frozen") == mark and cached["daily_sha256"] == digest(daily):
            return dest
    start = time.perf_counter()
    pnl, ratios, gate = predictions(model, job, data)
    ledger = books(paths, ratios, gate, job.cost)
    check = np.array([b[-1]["wealth"]/(float(p[0]["spot"])*int(p[0]["unit"])) for p,b in zip(paths,ledger)])
    error = float(np.max(np.abs(pnl-check)))
    if error > 1e-8:
        raise AssertionError(f"Differentiable ledger mismatch for {job.key}: {error:g}")
    seconds = record.get("seconds", 0.)
    dependencies = 0.
    if job.policy in GATES | {"static"}:
        dependencies = sum(json.loads(state_file(root, j).with_suffix(".json").read_text(
            encoding="utf8")).get("seconds", 0.) for j in expert_jobs(job).values())
    meta = dict(**job.meta(), registered=tag is None, frozen=mark, ledger_error=error,
                parameters=sum(p.numel() for p in model.parameters()), seconds=seconds,
                standalone_seconds=seconds+dependencies, replay_seconds=time.perf_counter()-start,
                date_to_xi=dict(zip(inputs.common["real_dates"], inputs.common["real_xi"])),
                reused=record.get("reused", False), updates=record.get("updates"))
    if tag is not None:
        meta["policy"] = tag
    return export(out, key, paths, ledger, job.cost, meta)


def statics(root, out, inputs, job, source):
    # Both fixed and selected weights use the same actual inventory and wealth.
    if not all(state_file(root, j).is_file() for j in expert_jobs(job).values()):
        return
    job = replace(job, policy="static", family="bridge")
    identity = dict(**source, experts={k: digest(state_file(root,j)) for k,j in expert_jobs(job).items()})
    _, val = inputs.observed(inputs.folds[1])
    scores = [score(predictions(restore(root,job,x/10),job,val)[0],job.lam) for x in range(11)]
    chosen = float(np.argmin(scores)/10)
    write_json((root/"policies"/f"{job.key}.json"), dict(identity=identity, score=scores, weight=chosen))
    for name, weight in (("fixed50", .5), ("static", chosen)):
        replay(root, out, inputs, job, dict(identity=identity), fixed=weight, tag=name)


def sensitivities(paths, inputs):
    dates = {d:i for i,d in enumerate(inputs.common["real_dates"])}
    rows = [row for path in paths for row in path[:-1]]
    i = np.array([dates[r["date"]] for r in rows])
    s = np.array([float(r["spot"]) for r in rows])
    k = np.array([float(r["strike"]) for r in rows])
    tau = (np.array([r["expiry"] for r in rows], dtype="datetime64[D]")-
           np.array([r["date"] for r in rows], dtype="datetime64[D]")).astype(float)/365
    c, dc, df = curve(tau, k/s, inputs.common["real_xi"][i], inputs.rep, inputs.common["scale"])
    d = c-(k/s)*dc
    cov = inputs.common["real_cov"][i]
    assert np.isfinite(cov).all() and (cov[:,0,0] > 1e-12).all()
    adjustment = np.sum(df*cov[:,1:,0], axis=1)/cov[:,0,0]
    assert np.isfinite(d+adjustment).all()
    cut = np.cumsum([len(path)-1 for path in paths])[:-1]
    return tuple(np.split(x, cut) for x in (d, adjustment))


def baselines(out, inputs, context, published):
    end, r, rep, market = context
    if market != "single":
        return
    paths = inputs.paths[end]
    ds, adjust = sensitivities(paths, inputs)
    for cost in (0,5,10) if end == 2025 else (5,):
        methods = {"Curve Delta": [np.clip(x,0,1) for x in ds],
                   "Local MV corrected": [np.clip(d+a,0,1) for d,a in zip(ds,adjust)]}
        if r == 0 and rep == "C0":
            methods["BS Delta"] = [bs_path(p) for p in paths]
        for name, ratios in methods.items():
            ledger = baseline_book(paths, ratios, cost)
            if name != "BS Delta":
                for book, delta, adj in zip(ledger, ds, adjust):
                    for i,row in enumerate(book):
                        row["model_delta"] = float(delta[i]) if i < len(delta) else ""
                        row["covariance_correction"] = float(adj[i]) if i < len(adj) else ""
                        row["unclipped_ratio"] = (float(delta[i]+(adj[i] if name == "Local MV corrected" else 0))
                                                  if i < len(adj) else "")
            meta = dict(end=end,r=r,rep=rep,t="off",w=False,h=False,cost=cost,market="baseline",policy=name,
                        family="baseline",lam=-1,seed=-1,registered=False,parameters=0,seconds=0,
                        standalone_seconds=0,replay_seconds=0,
                        date_to_xi=dict(zip(inputs.common["real_dates"],inputs.common["real_xi"])))
            export(out, f"{end}_R{r}_{rep}_{cost}bp_{name.replace(' ','_')}", paths, ledger, cost, meta)
    if r:
        return
    # Retain exact legacy ledgers with the same paths and a separate label.
    catalog = json.loads((published/"paths_index.json").read_text(encoding="utf8"))
    for x in catalog:
        if x["end"] != end or x["rep"] != rep:
            continue
        role = None
        if x["policy"] == "Local MV":
            role = "Local MV legacy"
        elif x["rep"] == "C0" and x["market"] == "gbm" and x["policy"] == "base":
            role = "GBM DH legacy"
        elif x["rep"] == "C0" and x["market"] == "single" and x["policy"] in {"base","factor"}:
            role = {"base":"Neural SDE DH legacy","factor":"Neural SDE DH factors legacy"}[x["policy"]]
        if role is None:
            continue
        name = x["id"]
        old = json.loads((published/"paths"/f"{name}.json").read_text(encoding="utf8"))
        assert [row["episode"] for row in old["rows"]] == [p[0]["episode"] for p in paths]
        source = published/"daily"/f"{name}.csv.gz"
        dest = out/"daily"/f"legacy_{name}.csv.gz"
        if not dest.exists():
            os.link(source, dest)
        assert digest(dest) == old["daily_sha256"]
        old["meta"].update(r=0,t="off",w=False,h=False,registered=False,family="baseline",policy=role)
        write_json(out/"paths"/f"legacy_{name}.json", old)


````

### 程序/extended_report.py

````python
"""Summarize frozen replays; all inference is paired at the expiry-month level."""
import csv
from collections import defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np


META = ("end", "cost", "lam", "rep", "market", "policy", "r", "t", "w", "h", "family")
MEASURES = ("rmse", "cvar95", "mean_pnl", "trade_cost", "turnover")
NUMBERS = (*MEASURES, "std", "parameters", "seconds", "standalone_seconds", "replay_seconds")
REPORT_VERSION = 1


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def normalized(meta):
    result = dict(meta)
    for k, default in (("r", 0), ("t", "off"), ("w", False), ("h", False), ("family", "baseline")):
        result.setdefault(k, default)
    for k in ("end", "r", "seed"):
        result[k] = int(result.get(k, -1))
    for k in ("cost", "lam"):
        result[k] = float(result.get(k, -1))
    for k in ("w", "h"):
        result[k] = str(result[k]).lower() in {"true", "1"}
    return result


def name(meta, legacy=False):
    prefix = "legacy/" if legacy else ""
    return prefix + "/".join(str(meta[k]) for k in META[3:])


def expected_shortfall(loss):
    loss = np.sort(np.asarray(loss), axis=-1)[..., ::-1]
    mass = .05 * loss.shape[-1]
    n = int(np.ceil(mass))
    return (loss[..., :n] * np.clip(mass - np.arange(n), 0, 1)).sum(-1) / mass


def measure(values):
    pnl, cost, turnover = np.moveaxis(values, -1, 0)
    return np.stack((np.sqrt(np.mean(pnl * pnl, axis=-1)), expected_shortfall(-pnl),
                     pnl.mean(-1), cost.mean(-1), turnover.mean(-1)), axis=-1)


def load_records(files, legacy=False):
    records = []
    for file in files:
        payload = json.loads(file.read_text(encoding="utf8"))
        meta = normalized(payload["meta"])
        rows = sorted(payload["rows"], key=lambda row: str(row["episode"]))
        if not rows:
            raise ValueError(f"Empty replay: {file.name}")
        ids = [str(row["episode"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate path IDs: {file.name}")
        values = np.array([[row[k] for k in ("pnl", "cost", "turnover")] for row in rows], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite replay: {file.name}")
        records.append(dict(meta=meta, rows=rows, values=values, ids=ids,
                            months=np.array([row["expiry"][:7] for row in rows]),
                            file=file, legacy=legacy, method=name(meta, legacy)))
    return records


def summarize(records):
    seeds, catalog = [], []
    for record in records:
        m, rows, values = record["meta"], record["rows"], record["values"]
        groups = {"all": np.ones(len(rows), dtype=bool)}
        for key in ("term", "moneyness", "cross_div", "adjusted", "expiry"):
            labels = np.array([str(row[key])[:7] if key == "expiry" else str(row[key]) for row in rows])
            for label in np.unique(labels):
                groups[f"{key}={label}"] = labels == label
        for group, mask in groups.items():
            row = {k: m[k] for k in META}
            row.update(method=record["method"], seed=m["seed"], group=group, n=int(mask.sum()),
                       registered=bool(m.get("registered", False)))
            row.update(zip(MEASURES, map(float, measure(values[mask]))))
            row["std"] = float(values[mask, 0].std())
            for key in NUMBERS[len(MEASURES) + 1:]:
                row[key] = float(m.get(key, m.get("seconds", 0) if key == "standalone_seconds" else 0))
            gates = [float(x["gate_mean"]) for x, keep in zip(rows, mask) if keep and x.get("gate_mean", "") != ""]
            row["gate_mean"] = float(np.mean(gates)) if gates else ""
            seeds.append(row)
        catalog.append(dict(id=record["file"].stem, method=record["method"],
                            **{k: m[k] for k in (*META, "seed")},
                            daily=f"daily/{record['file'].stem}.csv.gz", paths=f"paths/{record['file'].name}"))
    grouped = defaultdict(list)
    for row in seeds:
        grouped[tuple(row[k] for k in (*META, "method", "group"))].append(row)
    summary = []
    for keys, group in grouped.items():
        row = dict(zip((*META, "method", "group"), keys), seeds=len(group), n=group[0]["n"])
        if len({x["n"] for x in group}) != 1 or len({x["seed"] for x in group}) != len(group):
            raise ValueError(f"Inconsistent seeds or sample counts: {keys}")
        for key in NUMBERS:
            values = [x[key] for x in group]
            row.update({key: float(np.mean(values)), key + "_min": min(values), key + "_max": max(values)})
        summary.append(row)
    return seeds, summary, catalog


def strategy_arrays(records):
    grouped = defaultdict(dict)
    meta = {}
    for r in records:
        m = r["meta"]
        key = (m["end"], m["cost"], m["lam"], r["method"])
        if m["seed"] in grouped[key]:
            raise ValueError(f"Duplicate replay identity: {key}, seed={m['seed']}")
        grouped[key][m["seed"]] = r
        meta[key] = m
    arrays = {}
    for key, seeds in grouped.items():
        if set(seeds) not in ({0, 1, 2}, {-1}):
            continue  # Partial seeds are reported, never promoted to a complete contrast.
        order = sorted(seeds)
        first = seeds[order[0]]
        if any(r["ids"] != first["ids"] or not np.array_equal(r["months"], first["months"]) for r in seeds.values()):
            raise ValueError(f"Unpaired random seeds: {key}")
        arrays[key] = (np.stack([seeds[s]["values"] for s in order]), first["ids"], first["months"])
    return arrays, meta


def contrasts(arrays, metadata):
    """Build comparisons from registered cells, without selecting by outcomes."""
    result, seen = [], set()

    def add(kind, label, coefficients):
        coefficients = {k: v for k, v in coefficients.items() if v}
        if not coefficients or not all(k in arrays for k in coefficients):
            return
        keys = list(coefficients)
        first = arrays[keys[0]]
        if any(arrays[k][1] != first[1] or not np.array_equal(arrays[k][2], first[2]) for k in keys):
            raise ValueError(f"Different path IDs in comparison: {label}")
        token = (kind, tuple(sorted(coefficients.items())))
        if token in seen:
            return
        seen.add(token)
        end, cost, lam = next(k[:3] for k in keys if k[2] >= 0)
        result.append(dict(end=end, cost=cost, lam=lam, kind=kind, contrast=label,
                           n=len(first[1]), coefficients=coefficients))

    current = [k for k in arrays if not k[3].startswith("legacy/") and k[2] >= 0]
    by_root = defaultdict(list)
    for key in current:
        by_root[key[:3]].append(key)
    for root, keys in by_root.items():
        main = [k for k in keys if metadata[k]["family"] in {"main", "cost"} and metadata[k]["policy"] == "wide"
                and metadata[k]["market"] == "single" and metadata[k]["rep"] in {"C0", "C1", "C2"}
                and metadata[k]["t"] in {"off", "anneal"}]
        cells = {(metadata[k]["rep"], metadata[k]["r"], int(metadata[k]["t"] == "anneal"),
                  int(metadata[k]["w"]), int(metadata[k]["h"])): k for k in main}
        modules = ("representation", "R", "T", "W", "H")
        for a, b in combinations(sorted(cells), 2):
            add("factorial_pair", f"{b} minus {a}", {cells[b]: 1., cells[a]: -1.})
        for j, module in enumerate(modules[1:], 1):
            edges = [(cell, tuple(1 if i == j else v for i, v in enumerate(cell))) for cell in cells if not cell[j]]
            edges = [(a, b) for a, b in edges if b in cells]
            if edges:
                coef = defaultdict(float)
                for a, b in edges:
                    coef[cells[b]] += 1 / len(edges)
                    coef[cells[a]] -= 1 / len(edges)
                add("main_effect", f"{module}; average {len(edges)} matched edges", dict(coef))
            for rep in ("C0", "C1", "C2"):
                full = (rep, 1, 1, 1, 1)
                without = tuple(0 if i == j else v for i, v in enumerate(full))
                if full in cells and without in cells:
                    add("leave_one_out", f"{rep} full minus without {module}", {cells[full]: 1., cells[without]: -1.})
        for i, j in combinations(range(1, 5), 2):
            squares = []
            for cell in cells:
                if cell[i] or cell[j]:
                    continue
                corners = [tuple((1 if k in active else v) for k, v in enumerate(cell))
                           for active in ((), (i,), (j,), (i, j))]
                if all(c in cells for c in corners):
                    squares.append(corners)
            if squares:
                coef = defaultdict(float)
                for square in squares:
                    for c, sign in zip(square, (1, -1, -1, 1)):
                        coef[cells[c]] += sign / len(squares)
                add("interaction", f"{modules[i]} x {modules[j]}; average {len(squares)} squares", dict(coef))
        # The three-level factor representation needs explicit pairwise contrasts.
        for low, high in combinations(("C0", "C1", "C2"), 2):
            edges = [(cell, (high, *cell[1:])) for cell in cells if cell[0] == low and (high, *cell[1:]) in cells]
            if edges:
                coef = defaultdict(float)
                for a, b in edges:
                    coef[cells[b]] += 1 / len(edges)
                    coef[cells[a]] -= 1 / len(edges)
                add("main_effect", f"{high} minus {low}; average {len(edges)} matched edges", dict(coef))
            full, without = (high, 1, 1, 1, 1), (low, 1, 1, 1, 1)
            if full in cells and without in cells:
                add("leave_one_out", f"full {high} minus {low}", {cells[full]: 1., cells[without]: -1.})
            for j in range(1, 5):
                squares = []
                for a, b in edges:
                    if a[j]:
                        continue
                    aa = tuple(1 if i == j else v for i, v in enumerate(a))
                    bb = tuple(1 if i == j else v for i, v in enumerate(b))
                    if aa in cells and bb in cells:
                        squares.append((a, b, aa, bb))
                if squares:
                    coef = defaultdict(float)
                    for square in squares:
                        for c, sign in zip(square, (1, -1, -1, 1)):
                            coef[cells[c]] += sign / len(squares)
                    add("interaction", f"({high}-{low}) x {modules[j]}; average {len(squares)} squares", dict(coef))
        # Holding every other feature fixed isolates individual mechanisms.
        for a in keys:
            ma = metadata[a]
            for b in keys:
                if a == b:
                    continue
                mb = metadata[b]
                equal = lambda fields: all(ma[f] == mb[f] for f in fields)
                state = ("rep", "market", "r", "t", "w", "h")
                if equal(("rep", "market", "policy", "r", "w", "h")) and ma["t"] != mb["t"] and ma["t"] != "off" and mb["t"] in {"off", "anneal"}:
                    add("teacher", f"{a[3]} minus {b[3]}", {a: 1., b: -1.})
                if equal(state) and ma["policy"] in {"narrow", "flat", "gru", "delta_input", "delta_residual"} and mb["policy"] == "wide":
                    kind = "capacity" if ma["policy"] == "narrow" else "history" if ma["policy"] in {"flat", "gru"} else "delta"
                    add(kind, f"{a[3]} minus {b[3]}", {a: 1., b: -1.})
                if equal(("rep", "market", "r", "t", "w", "h")) and ma["policy"] == "gru" and mb["policy"] == "flat":
                    add("history", f"{a[3]} minus {b[3]}", {a: 1., b: -1.})
                if equal(("rep", "market", "r", "t", "w", "h")) and ma["policy"] == "delta_residual" and mb["policy"] == "delta_input":
                    add("delta", f"{a[3]} minus {b[3]}", {a: 1., b: -1.})
                if equal(("rep", "market", "r", "t", "w", "h")) and ma["policy"] in {"state_gate", "reliable_gate"} and mb["policy"] in {"wide", "capacity", "fixed50", "static", "base", "state_gate"}:
                    add("A", f"{a[3]} minus {b[3]}", {a: 1., b: -1.})
                if equal(("rep", "policy", "r", "t", "w", "h")) and ma["market"] == "robust" and mb["market"] in {"single", "equal"}:
                    add("B", f"{a[3]} minus {b[3]}", {a: 1., b: -1.})
                if equal(("market", "policy", "r", "t", "w", "h")) and ma["rep"] == "C2" and mb["rep"] in {"C0", "C1"}:
                    add("C", f"{a[3]} minus {b[3]}", {a: 1., b: -1.})
            # Strong controls retained from the prior study, including its best wide network.
            for b in arrays:
                mb = metadata[b]
                if b[0:2] != root[0:2] or b[2] not in (root[2], -1):
                    continue
                legacy = b[3].startswith("legacy/")
                reference = (mb["market"] == "single" and (mb["rep"], mb["policy"]) in {
                    ("C0", "factor"), ("C0", "same_capacity"), ("C1", "same_capacity"), ("C2", "factor")})
                baseline = ((not legacy and mb["family"] == "baseline")
                            or mb["policy"] in {"BS Delta", "Local MV"} or mb["market"] == "gbm")
                if (legacy and reference) or baseline:
                    add("legacy" if legacy else "baseline", f"{a[3]} minus {b[3]}", {a: 1., b: -1.})
    return result


def month_draws(months, length, repetitions):
    labels = np.unique(months)
    n = len(labels)
    rng = np.random.default_rng(1105)
    length = min(length, n)
    starts = rng.integers(n - length + 1, size=(repetitions, int(np.ceil(n / length))))
    indices = (starts[..., None] + np.arange(length)).reshape(repetitions, -1)[:, :n]
    counts = np.zeros((repetitions, n), dtype=np.int16)
    for j in range(n):
        counts[:, j] = np.sum(indices == j, axis=1)
    return counts, np.searchsorted(labels, months)


def boot_metrics(values, months, length, repetitions):
    """Weighted draws avoid sorting the same losses 2,000 times per seed."""
    counts, month_index = month_draws(months, length, repetitions)
    output = np.zeros((repetitions, len(MEASURES)))
    for sample in values:
        loss_order = np.argsort(-sample[:, 0])[::-1]
        losses = -sample[loss_order, 0]
        for start in range(0, repetitions, 128):
            weights = counts[start:start + 128, month_index].astype(float)
            n = weights.sum(1)
            squared = weights @ (sample[:, 0] ** 2) / n
            tail_weights = weights[:, loss_order]
            before = np.cumsum(tail_weights, axis=1) - tail_weights
            tail = np.clip(.05 * n[:, None] - before, 0, tail_weights)
            es = tail @ losses / (.05 * n)
            means = weights @ sample / n[:, None]
            output[start:start + 128] += np.column_stack((np.sqrt(squared), es, means)) / len(values)
    return output


def infer(out, arrays, definitions, repetitions, identity):
    needed = sorted({k for item in definitions for k in item["coefficients"]})
    cache_dir = out / "bootstrap_cache"
    cache_dir.mkdir(exist_ok=True)
    seed_points = {k: measure(arrays[k][0]) for k in needed}
    points = {k: v.mean(0) for k, v in seed_points.items()}
    results = []
    for length in (1, 3, 6):
        cache = {}
        for key in needed:
            token = hashlib.sha256(json.dumps([identity, key, length, repetitions], sort_keys=True).encode()).hexdigest()
            path = cache_dir / (token + ".npz")
            if path.exists():
                with np.load(path) as saved:
                    cache[key] = saved["draws"]
            else:
                values, _, months = arrays[key]
                cache[key] = boot_metrics(values, months, length, repetitions)
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, draws=cache[key])
                temporary.replace(path)
        for item in definitions:
            coef = item["coefficients"]
            point = sum(points[k] * v for k, v in coef.items())
            seed_delta = sum(seed_points[k] * v for k, v in coef.items())
            draws = sum(cache[k] * v for k, v in coef.items())
            lo, hi = np.quantile(draws, [.025, .975], axis=0)
            for j, metric in enumerate(MEASURES):
                row = {k: v for k, v in item.items() if k != "coefficients"}
                row.update(block_months=length, metric=metric, delta=float(point[j]), low=float(lo[j]), high=float(hi[j]),
                           terms=json.dumps({k[3]: v for k, v in coef.items()}, ensure_ascii=False),
                           repetitions=repetitions, seed_unit="mean_of_seedwise_metrics",
                           delta_seed_min=float(seed_delta[:, j].min()), delta_seed_max=float(seed_delta[:, j].max()),
                           favorable_seeds=int(np.sum(seed_delta[:, j] > 0 if metric == "mean_pnl" else seed_delta[:, j] < 0)),
                           seed_count=len(seed_delta))
                results.append(row)
    return results


def plots(out, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in summary if r["group"] == "all" and r["end"] == 2025 and r["cost"] == 5 and r["lam"] == 0]
    fig, ax = plt.subplots(figsize=(9, 6))
    families = sorted({r["family"] for r in rows})
    for family in families:
        subset = [r for r in rows if r["family"] == family]
        ax.scatter([100 * r["rmse"] for r in subset], [100 * r["cvar95"] for r in subset], label=family, alpha=.7)
    ax.set(xlabel="RMSE (%)", ylabel="CVaR95 loss (%)", title="All registered candidates: historical 2024-2025, 5bp, MSE")
    if families:
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "risk_tradeoff.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ("rmse", "cvar95")):
        for rep in ("C0", "C1"):
            for full in (False, True):
                selected = sorted([r for r in summary if r["group"] == "all" and r["cost"] == 5 and r["lam"] == 0
                                   and r["rep"] == rep and r["market"] == "single" and r["policy"] == "wide"
                                   and r["r"] == int(full) and r["w"] == full and r["h"] == full
                                   and r["t"] == ("anneal" if full else "off")], key=lambda r: r["end"])
                ax.plot([r["end"] for r in selected], [100 * r[metric] for r in selected], marker="o", label=rep + (" full" if full else " plain"))
        ax.set(xlabel="Evaluation end year", ylabel=metric + " (%)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "rolling_stability.png", dpi=160); plt.close(fig)


def dashboard(out, summary, catalog, completion):
    # Embedded summaries work via file://. Detailed saved books are local links.
    data = json.dumps(dict(summary=summary, paths=catalog, completion=completion), ensure_ascii=False).replace("</", "<\\/")
    page = '''<!doctype html><meta charset="utf-8"><title>50ETF 扩展实验</title>
<style>body{font:15px system-ui;margin:32px;background:#f7f8fa;color:#18232b}select,input{padding:7px;margin:5px}table{border-collapse:collapse;background:white;width:100%}td,th{padding:8px;text-align:right;border-bottom:1px solid #ddd}td:first-child,th:first-child{text-align:left}.scroll{max-height:65vh;overflow:auto}th{position:sticky;top:0;background:#e9edf1}small{color:#556}</style>
<h1>50ETF 扩展组合实验</h1><p>2024–2025 为已观察历史评价。指标越小越好，但平均损益越大越好。种子不是额外市场样本。</p><p id="status"></p>
<label>截止年 <select id="end"></select></label><label>成本 <select id="cost"></select></label><label>目标 <select id="lam"></select></label><label>分组 <select id="group"></select></label><input id="query" placeholder="组合、表示或策略名称"><label>排序 <select id="metric"><option>rmse</option><option>cvar95</option><option>trade_cost</option><option>standalone_seconds</option></select></label>
<div class="scroll"><table><thead><tr><th>方法</th><th>样本/种子</th><th>RMSE</th><th>CVaR95</th><th>平均损益</th><th>成本</th><th>耗时(秒)</th></tr></thead><tbody id="rows"></tbody></table></div>
<h2>已保存的具体路径和逐日持仓</h2><small>链接只读取本地保存结果，不触发重新训练。路径 JSON 可查看合约编号和单条损益；逐日压缩账本含持仓、现金和成本。</small><div id="books"></div>
<p><a href="paired_intervals.csv">配对区间</a> · <a href="factorial_contrasts.csv">主效应与交互</a> · <a href="legacy_comparison.csv">历史强对照</a> · <a href="failures.json">失败与缺项</a></p>
<script>const DATA=__DATA__;const $=x=>document.getElementById(x);const escape=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
for(const k of ['end','cost','lam','group']){let v=[...new Set(DATA.summary.map(r=>r[k]))].sort((a,b)=>typeof a==='number'?a-b:String(a).localeCompare(String(b)));$(k).innerHTML=v.map(x=>`<option>${escape(x)}</option>`).join('');} $('end').value=2025;$('cost').value=5;$('lam').value=0;$('group').value='all';$('status').textContent=JSON.stringify(DATA.completion);
function draw(){let selected=DATA.summary.filter(r=>['end','cost','lam','group'].every(k=>String(r[k])===$(k).value)&&r.method.toLowerCase().includes($('query').value.toLowerCase())).sort((a,b)=>a[$('metric').value]-b[$('metric').value]);const pct=x=>(100*x).toFixed(4)+'%';$('rows').innerHTML=selected.map(r=>`<tr><td>${escape(r.method)}</td><td>${r.n}/${r.seeds}</td><td>${pct(r.rmse)}</td><td>${pct(r.cvar95)}</td><td>${pct(r.mean_pnl)}</td><td>${pct(r.trade_cost)}</td><td>${r.standalone_seconds.toFixed(1)}</td></tr>`).join('');const names=new Set(selected.map(r=>r.method));$('books').innerHTML=DATA.paths.filter(r=>names.has(r.method)&&String(r.end)===$('end').value&&String(r.cost)===$('cost').value&&String(r.lam)===$('lam').value).map(r=>`<p>${escape(r.method)} / seed ${r.seed} <a href="${encodeURI(r.paths)}">逐路径结果</a> <a href="${encodeURI(r.daily)}">逐日账本</a></p>`).join('');}for(const k of ['end','cost','lam','group','query','metric'])$(k).oninput=draw;draw();</script>'''
    (out / "index.html").write_text(page.replace("__DATA__", data), encoding="utf8")


def report(root, project, cfg, failures, expected=1776):
    root, project = Path(root), Path(project)
    out = root / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((out / "paths").glob("*.json"))
    if not files:
        raise ValueError("No frozen evaluation paths available")
    records = load_records(files)
    frozen_path = out / "frozen_checkpoints.json"
    stale = []
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf8"))
        if not isinstance(frozen, dict):
            raise ValueError("Expected checkpoint manifest mapping job keys to hashes")
        stale = [r for r in records if r["meta"].get("registered", False) and
                 (r["file"].stem not in frozen or (r["meta"].get("frozen") and
                  r["meta"]["frozen"].get("weights") != frozen.get(r["file"].stem)))]
        stale_paths = {r["file"] for r in stale}
        records = [r for r in records if r["file"] not in stale_paths]
    # Ensure all strategies in a fold use exactly the same evaluation episodes.
    fold_ids = {}
    for r in records:
        end = r["meta"]["end"]
        if end in fold_ids and fold_ids[end] != r["ids"]:
            raise ValueError(f"Unfair evaluation samples in fold {end}")
        fold_ids[end] = r["ids"]
    seeds, summary, catalog = summarize(records)
    write_csv(out / "metrics_by_seed.csv", seeds)
    write_csv(out / "summary.csv", summary)
    write_json(out / "paths_index.json", catalog)
    old = project / "交付区" / "结果" / "创新组合实验"
    legacy = []
    if (old / "summary.csv").exists():
        with (old / "summary.csv").open(encoding="utf8", newline="") as stream:
            previous = list(csv.DictReader(stream))
        write_csv(out / "legacy_comparison.csv", [dict(source="previous", **r) for r in previous]
                  + [dict(source="extension", **r) for r in summary])
        candidates = [f for f in (old / "paths").glob("*.json") if any(s in f.name for s in
                      ("same_capacity", "_factor.json", "_gbm_", "_BS_Delta.json", "_Local_MV.json"))]
        legacy = load_records(candidates, legacy=True)
    else:
        write_csv(out / "legacy_comparison.csv", [dict(source="extension", **r) for r in summary])
    arrays, metadata = strategy_arrays(records + legacy)
    definitions = contrasts(arrays, metadata)
    bootstrap = cfg.get("bootstrap", 2000)
    repetitions = int(bootstrap.get("repetitions", 2000) if isinstance(bootstrap, dict) else bootstrap)
    identity = hashlib.sha256((str(REPORT_VERSION) + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                              + json.dumps(cfg, sort_keys=True, ensure_ascii=False)
                              + "".join(hashlib.sha256(r["file"].read_bytes()).hexdigest() for r in records + legacy)).encode()).hexdigest()
    pairs = infer(out, arrays, definitions, repetitions, identity)
    write_csv(out / "paired_intervals.csv", pairs)
    write_csv(out / "factorial_contrasts.csv", [r for r in pairs if r["kind"] in {"main_effect", "interaction", "leave_one_out"}])
    write_csv(out / "mechanism_contrasts.csv", [r for r in pairs if r["kind"] in {"teacher", "capacity", "history", "delta", "A", "B", "C"}])
    registered = {tuple(r["meta"][k] for k in (*META, "seed")) for r in records if r["meta"].get("registered", False)}
    failure_rows = list(failures)
    unavailable = sum(f.get("status") == "unavailable" for f in failure_rows)
    failed = sum(f.get("status") not in {"unavailable", "baseline_unavailable"} for f in failure_rows)
    completion = dict(expected=int(expected), evaluated=len(registered), unavailable=unavailable, failures=failed,
                      baseline_unavailable=sum(f.get("status") == "baseline_unavailable" for f in failure_rows),
                      missing=max(0, int(expected) - len(registered) - unavailable - failed),
                      replay_files=len(records), complete=len(registered) + unavailable == expected and failed == 0,
                      excluded_stale_replays=len(stale),
                      paired_definitions=len(definitions), report_identity=identity,
                      historical_evaluation="2024-2025 already observed", repetitions=repetitions)
    write_json(out / "completion.json", completion)
    write_json(out / "failures.json", failure_rows)
    write_json(out / "report_manifest.json", dict(identity=identity, version=REPORT_VERSION,
               files=len(files), bootstrap_seed=1105, blocks=[1, 3, 6], samples={str(k): len(v) for k, v in fold_ids.items()}))
    visual_summary = list(summary)
    visual_catalog = list(catalog)
    if legacy:
        _, legacy_summary, legacy_catalog = summarize(legacy)
        visual_summary.extend(r for r in legacy_summary if r["group"] == "all" and
                              r["market"] == "single" and r["policy"] == "same_capacity")
        for item in legacy_catalog:
            if item["market"] == "single" and item["policy"] == "same_capacity":
                item["daily"] = old.resolve().as_uri() + "/" + item["daily"]
                item["paths"] = old.resolve().as_uri() + "/" + item["paths"]
                visual_catalog.append(item)
        for r in visual_summary:
            if r["method"].startswith("legacy/"):
                r["family"] = "legacy wide controls"
    plots(out, visual_summary)
    dashboard(out, visual_summary, visual_catalog, completion)
    rows = sorted([r for r in summary if r["group"] == "all" and (r["end"], r["cost"], r["lam"]) == (2025, 5, 0)], key=lambda r: r["rmse"])
    lines = ["# 50ETF 扩展组合实验结果", "", f"登记训练评价完成 {len(registered)}/{expected}；不可运行 {unavailable}，失败 {failed}，待补 {completion['missing']}。", "",
             "2024–2025 是已观察的历史评价。下表包含强普通网络和所有可用组合，不把预设核心组合误称为全体最优。完整分组、种子范围、机制比较和历史强对照见 CSV 与本地展示页面。", "",
             "| 方法 | 样本/种子 | RMSE | CVaR95 损失 | 交易成本 |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['n']}/{r['seeds']} | {r['rmse']:.4%} | {r['cvar95']:.4%} | {r['trade_cost']:.4%} |")
    focus = [r for r in pairs if r["end"] == 2025 and r["cost"] == 5 and r["lam"] == 0
             and r["kind"] in {"main_effect","interaction"} and r["block_months"] == 3
             and r["metric"] in {"rmse","cvar95"}]
    if focus:
        lines += ["", "## 5bp 主时期的模块效应", "",
                  "| 预登记对照 | 指标 | 差值（百分点） | 95% 区间 | 同方向改善的滚动折 |",
                  "|---|---|---:|---:|---:|"]
        for row in focus:
            same = [p for p in pairs if p["end"] in (2021,2022,2023)
                    and all(p[k] == row[k] for k in ("cost","lam","kind","contrast","block_months","metric"))]
            interval = f"[{100*row['low']:+.4f}, {100*row['high']:+.4f}]"
            if row["low"] <= 0 <= row["high"]:
                interval += "（证据有限）"
            improving = sum(x["delta"] < 0 for x in same)
            lines.append(f"| {row['contrast']} | {row['metric']} | {100*row['delta']:+.4f} | {interval} | {improving}/{len(same)} |")
    lines += ["", "差值为候选减对照。RMSE、尾部损失和成本差值为负表示下降；平均损益差值为正表示增加。主效应平均所有匹配边，二阶交互为双模块效应减两个单模块效应，移除对照为完整组合减移除一个模块。", "",
              "区间按相同到期月份配对，分别使用 1、3、6 个月连续块，固定 2,000 次（调试配置可降低）重抽样。先在各种子内计算指标，再平均；种子不扩充市场样本。各区间未做多重比较校正，跨零时写证据有限，不宣布唯一总冠军。", "",
              "统计显著性不能修复已经观察过历史评价集的问题。连续无分红模拟、同收盘观测成交、离散边界修正等近似继续适用。策略耗时不等于共享市场模型加策略的全部研究成本。"]
    (out / "结果说明.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    return completion

````

### 程序/extended_train.py

````python
"""Train frozen jobs with path-level risk and optional train-only guidance."""
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from contract_ledger import ledger
from prepare_common import digest, write_json
from train import adversary, batches, expected_shortfall, make_model, make_real, replay, score
from extended_data import batch
from extended_policy import load_legacy, make_policy, rollout, simulated_pnl


GATES = {"state_gate", "reliable_gate"}
DELTA = {"delta_input", "delta_residual"}


def needs_history(job):
    return job.h or job.policy in {"flat", "gru"}


def make_model_for(job, experts=None, fixed=None):
    return make_policy(job.policy, wealth=job.w, history=job.h, experts=experts, fixed=fixed)


def state_file(root, job):
    return root/"policies"/f"{job.key}.pt"


def expert_jobs(job):
    return {kind: replace(job, policy=kind) for kind in ("base", "wide")}


def restore(root, job, fixed=None):
    experts = None
    if job.policy in GATES | {"static"}:
        experts = {k: restore(root, j) for k, j in expert_jobs(job).items()}
    model = make_model_for(job, experts, fixed)
    if fixed is None:
        model.load_state_dict(torch.load(state_file(root, job), weights_only=True))
    return model


def risk_terms(z, lam, eta):
    loss = z.square()
    if lam:
        loss = loss + lam*(eta + torch.relu(-z-eta)/.05)
    return loss


def teacher_loss(ratio, data):
    mask = data["teacher_valid"]
    per_path = ((ratio-data["teacher"]).square()*mask).sum(1) / mask.sum(1).clamp(min=1)
    return per_path.sum() / mask.any(1).sum().clamp(min=1)


def norm(loss, parameters):
    return torch.stack([p.detach().square().sum() for p in
                        torch.autograd.grad(loss, parameters, allow_unused=True)
                        if p is not None]).sum().sqrt().item()


@torch.no_grad()
def predictions(model, job, data, size=512):
    pnl, ratios, gates = [], [], []
    for i in range(0, len(data["f"]), size):
        x = batch(data, slice(i, i+size), needs_history(job))
        p, r, g = rollout(model, x, job.cost/10000, capture=True)
        pnl.append(p.detach().numpy())
        ratios.append(r.detach().numpy())
        if g is not None:
            gates.append(g.detach().numpy())
    return np.concatenate(pnl), np.concatenate(ratios), np.concatenate(gates) if gates else None


def fit(model, job, cfg, inputs):
    start = time.perf_counter()
    simulated, ids = inputs.simulation(job.market, job.policy in DELTA)
    n = cfg["simulation_train"]
    assert len(ids) == n + cfg["simulation_valid"]
    _, val = inputs.observed(inputs.folds[1], job.policy in DELTA)
    guidance = job.t != "off"
    if guidance:
        _, real = inputs.observed(inputs.folds[0], job.policy in DELTA, teacher=job.t != "real")
    eta = torch.nn.Parameter(torch.tensor(1.))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params+([eta] if job.lam else []),
                           lr=.001 if job.policy in GATES else .003)
    sim_rng = np.random.default_rng(job.seed)
    real_rng = np.random.default_rng(job.seed+8401)
    levels = np.unique(ids[:n])
    fee = job.cost/10000

    def sim_loss(idx):
        p = simulated_pnl(model, batch(simulated, idx, needs_history(job)), fee)
        terms = risk_terms(100*p, job.lam, eta)
        risks = torch.stack([terms[ids[idx] == j].mean() for j in levels])
        q = adversary(risks.detach().numpy(), .1 if job.market == "robust" else 0)
        return (risks*torch.as_tensor(q, dtype=risks.dtype)).sum(), risks, q

    def add_loss(idx):
        data = batch(real, idx, needs_history(job))
        p, r, _ = rollout(model, data, fee)
        return (risk_terms(100*p, job.lam, eta).mean() if job.t == "real"
                else teacher_loss(r, data))

    beta = 0.
    if guidance and job.t != "warm":
        sim_cal = np.random.default_rng(job.seed+548)
        real_cal = np.random.default_rng(job.seed+549)
        a, b = [], []
        for idx in list(batches(ids[:n], cfg["batch_size"], sim_cal))[:cfg["gradient_batches"]]:
            a.append(norm(sim_loss(idx)[0], params))
            b.append(norm(add_loss(real_cal.integers(len(real["f"]), size=cfg["real_batch_size"])), params))
        beta = float(np.clip(.1*np.mean(a)/max(np.mean(b), 1e-8), 1e-4, 1e4))

    best, stale, trace = float("inf"), 0, []
    updates = 0
    for epoch in range(1, cfg["epochs"]+1):
        train_risk, train_q = [], []
        pretrain = job.t == "warm" and epoch <= cfg["warm_epochs"]
        alpha = beta
        if job.t in {"anneal", "real"}:
            alpha *= max(0., 1-(epoch-1)/max(cfg["teacher_epochs"]-1, 1))
        for idx in batches(ids[:n], cfg["batch_size"], sim_rng):
            opt.zero_grad(set_to_none=True)
            if pretrain or alpha:
                ridx = real_rng.integers(len(real["f"]), size=cfg["real_batch_size"])
                extra = add_loss(ridx)
            if pretrain:
                loss = extra
            else:
                loss, risks, q = sim_loss(idx)
                train_risk.append(risks.detach().numpy())
                train_q.append(q)
                if alpha:
                    loss = loss + alpha*extra
                    if job.t == "real":
                        loss = loss/(1+alpha)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite objective at epoch {epoch}")
            loss.backward()
            opt.step()
            updates += 1
        if epoch % cfg["validation_every"]:
            continue
        p = predictions(model, job, val)[0]
        value = score(p, job.lam)
        if not np.isfinite(value):
            raise FloatingPointError(f"nonfinite validation at epoch {epoch}")
        with torch.no_grad():
            # Keep memory bounded and retain equal prior weight per market.
            pred = torch.cat([simulated_pnl(model, batch(simulated, slice(i,i+512), needs_history(job)), fee)
                              for i in range(n,len(ids),512)])
            terms = risk_terms(100*pred, job.lam, eta).numpy()
            risk = np.array([terms[ids[n:] == j].mean() for j in levels])
            q = adversary(risk, .1 if job.market == "robust" else 0)
        trace.append(dict(epoch=epoch, updates=updates, validation=value,
                          rmse=float(np.sqrt(np.mean(p*p))), cvar=float(expected_shortfall(-p)),
                          beta=alpha, eta=float(eta.detach()),
                          train_risk=np.mean(train_risk, axis=0).tolist() if train_risk else [],
                          train_q=np.mean(train_q, axis=0).tolist() if train_q else [],
                          sim_risk=risk.tolist(), sim_q=q.tolist(),
                          sim_kl=float(np.sum(q*np.log(np.maximum(q*len(q),1e-300)))),
                          effective_markets=float(1/np.sum(q*q))))
        if value < best:
            best, stale, best_epoch = value, 0, epoch
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    return state, dict(epoch=best_epoch, best=best, seconds=time.perf_counter()-start,
                       trace=trace, updates=updates, beta=beta,
                       teacher_valid=int(real["teacher_valid"].sum()) if guidance and job.t != "real" else None,
                       teacher_decisions=int(real["active"].sum()) if guidance and job.t != "real" else None)


def legacy(job, model, inputs, original, parent, episodes_sha, cfg):
    if job.r or job.w or job.h or job.t != "off" or job.policy not in {"wide", "narrow"}:
        return
    name = {"wide": "same_capacity", "narrow": "same_inputs"}[job.policy]
    group = original/str(job.end)/"policies"/f"{job.rep}_{job.market}_{job.cost}bp_l{job.lam:g}_s{job.seed}"
    weights = group/f"{name}.pt"
    status = group/f"{name}.json"
    if not weights.exists() or not status.exists():
        return
    r = json.loads(status.read_text(encoding="utf8"))
    source = r.get("identity", {})
    old_cfg = json.loads((original/str(job.end)/"config.json").read_text(encoding="utf8"))
    keys = ("simulation_train", "simulation_valid", "batch_size", "epochs",
            "validation_every", "patience", "folds", "seeds", "lambdas", "origin_seed", "shock_seed")
    if (r.get("status") != "complete" or source.get("parent") != parent
        or source.get("episodes") != episodes_sha or digest(weights) != r.get("weights_sha256")
        or any(cfg[k] != old_cfg[k] for k in keys)):
        return
    state = torch.load(weights, weights_only=True)
    load_legacy(model, state)
    old = make_model(name)
    old.load_state_dict(state)
    paths, data = inputs.observed(inputs.folds[1])
    old_pnl = replay(old, paths, make_real(paths, inputs.common, inputs.episode_file), job.cost/10000)[0]
    p, ratios, _ = predictions(model, job, data)
    compatible = np.array([float(ledger(path, ratios[i,:len(path)-1], float(path[0]["option_close"]),
                        job.cost/10000)[-1]["wealth"])/(float(path[0]["spot"])*int(path[0]["unit"]))
                        for i,path in enumerate(paths)])
    error = float(np.max(np.abs(compatible-old_pnl)))
    if error > 1e-8:
        raise AssertionError(f"Legacy replay mismatch {job.key}: {error}")
    return model.state_dict(), {**r, "legacy_error": error, "legacy_weights_sha256": digest(weights),
                                "legacy_account_precision_gap": float(np.max(abs(p-compatible))),
                                "legacy_best": r["best"], "best": score(p,job.lam), "reused": True}


def complete(root, job, identity):
    weights = state_file(root, job)
    status = weights.with_suffix(".json")
    if not weights.exists() or not status.exists():
        return False
    r = json.loads(status.read_text(encoding="utf8"))
    return (r.get("status") == "complete" and r.get("identity") == identity
            and r.get("weights_sha256") == digest(weights))


def train_job(root, job, cfg, inputs, original, source):
    torch.manual_seed(job.seed)
    weights = state_file(root, job)
    identity = {**source, "job": job.meta()}
    experts = None
    if job.policy in GATES | {"capacity"}:
        parents = expert_jobs(job)
        identity["experts"] = {}
        for k, parent in parents.items():
            checkpoint = state_file(root, parent)
            status = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf8"))
            sha = digest(checkpoint)
            if (status.get("status") != "complete" or status.get("weights_sha256") != sha
                    or any(status.get("identity",{}).get(key) != value for key,value in source.items())):
                raise ValueError(f"Expert must be complete in the current run: {parent.key}")
            identity["experts"][k] = sha
        experts = {k: restore(root, j) for k, j in parents.items()}
    if complete(root, job, identity):
        return "cached"
    model = make_model_for(job, experts)
    fitted = legacy(job, model, inputs, original, source["market"], source["episodes"], cfg)
    state, stats = fitted if fitted else fit(model, job, cfg, inputs)
    temp = weights.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(weights)
    model.load_state_dict(torch.load(weights, weights_only=True))
    _, data = inputs.observed(inputs.folds[1], job.policy in DELTA)
    value = score(predictions(model, job, data)[0], job.lam)
    if abs(value-stats["best"]) > (1e-6 if fitted else 1e-10):
        raise AssertionError(f"Saved model changed validation risk {value} vs {stats['best']}")
    write_json(weights.with_suffix(".json"), dict(**{k: v for k, v in stats.items() if k not in
               {"identity", "status", "parameters", "weights_sha256"}}, status="complete",
               identity=identity, parameters=sum(p.numel() for p in model.parameters()),
               weights_sha256=digest(weights)))
    return "reused" if fitted else "trained"


````

### 程序/interim_report.py

````python
"""Interim frozen main-period replay requested during long rolling training."""
import csv
import json
from pathlib import Path
from dataclasses import asdict
import time

import numpy as np
import torch

from prepare_common import digest,write_json
from extended_registry import registry
from extended_data import Inputs
from extended_train import predictions,restore,state_file
from extended_report import boot_metrics
from run_extended import sources
from train import expected_shortfall


def run(project):
    torch.set_num_threads(1)
    here=Path(__file__).resolve().parent
    root=project/"工作区/数据/处理后/扩展组合实验"
    cfg=json.loads((here/"study.json").read_text(encoding="utf8"))
    selected=[j for j in registry(cfg).values() if j.family=="main" and j.end==2025 and j.cost==5 and j.lam==0]
    assert len(selected)==144
    frozen={}
    for j in selected:
        path=state_file(root,j)
        r=json.loads(path.with_suffix(".json").read_text(encoding="utf8"))
        assert r["status"]=="complete" and digest(path)==r["weights_sha256"]
        frozen[j.key]=dict(job=j.meta(),sha256=digest(path),best_epoch=r["epoch"])
    out=root/"interim_20261108"
    out.mkdir(exist_ok=True)
    write_json(out/"frozen_checkpoints.json",frozen)
    groups={}
    ids=None
    months=None
    for r in (0,1):
        for rep in ("C0","C1","C2"):
            paths,_=sources(project,root,(2025,r,rep,"single"))
            inputs=Inputs(*paths,cfg["folds"][0])
            test,data=inputs.observed(2025)
            current=[p[0]["episode"] for p in test]
            assert len(test)==773 and (ids is None or ids==current)
            ids=current
            months=np.array([p[0]["expiry"][:7] for p in test])
            for j in selected:
                if (j.r,j.rep)!=(r,rep):continue
                assert digest(state_file(root,j))==frozen[j.key]["sha256"]
                pnl,_,_=predictions(restore(root,j),j,data)
                key=f"{j.rep}_R{j.r}_T{j.t}_W{int(j.w)}H{int(j.h)}"
                groups.setdefault(key,{})[j.seed]=pnl
            print("Replayed",r,rep,flush=True)
    values={k:np.stack([s[i] for i in (0,1,2)]) for k,s in groups.items()}
    np.savez_compressed(out/"path_pnl.npz",episode=np.array(ids),month=months,**values)
    points={k:np.stack([np.sqrt(np.mean(x*x,axis=1)),expected_shortfall(-x,axis=1)],axis=1) for k,x in values.items()}
    reference="C1_R0_Toff_W0H0"
    ref=points[reference].mean(0)
    draws={}
    for k,x in values.items():
        # Only the first two bootstrap columns are used. Cost/turnover are not estimated here.
        placeholder=np.stack([x,np.zeros_like(x),np.zeros_like(x)],axis=-1)
        draws[k]=boot_metrics(placeholder,months,3,2000)[:,:2]
    rows=[]
    for k,p in points.items():
        delta=p.mean(0)-ref
        lo,hi=np.quantile(draws[k]-draws[reference],[.025,.975],axis=0)
        row=dict(method=k,seeds=3,n=773,rmse=float(p[:,0].mean()),cvar95=float(p[:,1].mean()),
                 rmse_min=float(p[:,0].min()),rmse_max=float(p[:,0].max()),
                 cvar_min=float(p[:,1].min()),cvar_max=float(p[:,1].max()),
                 rmse_delta=float(delta[0]),cvar_delta=float(delta[1]),
                 rmse_low=float(lo[0]),rmse_high=float(hi[0]),cvar_low=float(lo[1]),cvar_high=float(hi[1]),
                 better_rmse_seeds=int(np.sum(p[:,0]<points[reference][:,0])),
                 better_cvar_seeds=int(np.sum(p[:,1]<points[reference][:,1])))
        rows.append(row)
    rows.sort(key=lambda row:row["rmse"])
    with (out/"summary_5bp_mse.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    write_json(out/"status.json",dict(stage="interim",models=144,configurations=48,cost_bp=5,lam=0,
               paths=773,seeds=[0,1,2],reference=reference,historical_evaluation="2024-2025 already observed",
               checkpoint_selection="unchanged validation selection",bootstrap=2000,block_months=3))
    for row in rows[:8]:print(json.dumps(row),flush=True)
    for k in (reference,"C1_R1_Tanneal_W1H1"):
        print("REFERENCE_OR_FULL",json.dumps(next(row for row in rows if row["method"]==k)),flush=True)


if __name__=="__main__":
    run(Path.cwd())

````

### 程序/launch_detached.py

````python
"""Launch the frozen experiment through Windows' independent process provider."""
import ctypes
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

def in_job(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.IsProcessInJob.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000,False,pid)
    flag = ctypes.c_int()
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not kernel.IsProcessInJob(handle,None,ctypes.byref(flag)):
            raise ctypes.WinError(ctypes.get_last_error())
        return bool(flag.value)
    finally:
        kernel.CloseHandle(handle)

def write(path,data):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf8")
    temporary.replace(path)

def main():
    project=Path.cwd()
    program=Path(__file__).resolve().parent
    out=project/"工作区/数据/处理后/扩展组合实验"
    import os

    old=out/"runner.json"
    if old.exists():
        previous=json.loads(old.read_text(encoding="utf-8-sig"))
        # Do not replace an experiment that is still alive.
        result=subprocess.run(["tasklist","/FI",f"PID eq {previous['pid']}","/FO","CSV","/NH"],
                              capture_output=True,text=True)
        if f'"{previous["pid"]}"' in result.stdout:
            raise RuntimeError("The previous experiment process is still alive")
    tag=datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    stdout=out/f"{tag}.stdout.log"
    stderr=out/f"{tag}.stderr.log"
    command=["powershell.exe","-NoProfile","-ExecutionPolicy","Bypass","-File",
             str(program/"run_extended.ps1"),"-OptionFile",
             str(project.parent/"开题/test/510050.XSHG_option_full.csv"),
             "-SpotFile","D:/life/study/quant/投资/数据/ETF/510050_华夏上证50ETF.csv",
             "-Workers","8","-Publish"]
    with stdout.open("wb") as output,stderr.open("wb") as error:
        process=subprocess.Popen(command,cwd=project,stdout=output,stderr=error,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
    record=dict(pid=process.pid,supervisor_pid=os.getpid(),stdout=str(stdout),stderr=str(stderr),
                start=datetime.now().astimezone().isoformat(),launch="Win32_Process independent provider",
                in_windows_job=in_job(process.pid))
    write(out/"runner.json",record)

    resource=out/f"{tag}.resources.csv"
    with (out/f"{tag}.monitor.stderr.log").open("wb") as error:
        subprocess.Popen(["powershell.exe","-NoProfile","-ExecutionPolicy","Bypass",
                          "-File",str(program/"monitor_resources.ps1"),"-RunnerId",str(process.pid),
                          "-Destination",str(resource)],cwd=project,stdout=subprocess.DEVNULL,stderr=error,
                          creationflags=subprocess.CREATE_NO_WINDOW)
    code=process.wait()
    record.update(exit_code=code,finished=datetime.now().astimezone().isoformat())
    write(out/"runner.json",record)

if __name__=="__main__":
    main()

````

### 程序/monitor_resources.ps1

````powershell
param(
    [Parameter(Mandatory=$true)][int]$RunnerId,
    [Parameter(Mandatory=$true)][string]$Destination
)
$ErrorActionPreference = 'Stop'
$runner = Get-Process -Id $RunnerId
$start = $runner.StartTime
while ($true) {
    $runner = Get-Process -Id $RunnerId -ErrorAction SilentlyContinue
    if (!$runner -or $runner.StartTime -ne $start) { break }
    $tree = @(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId)
    $ids = [Collections.Generic.HashSet[int]]::new()
    [void]$ids.Add($RunnerId)
    do {
        $count = $ids.Count
        foreach ($row in $tree) {
            if ($ids.Contains([int]$row.ParentProcessId)) {
                [void]$ids.Add([int]$row.ProcessId)
            }
        }
    } while ($ids.Count -gt $count)
    $processes = @(Get-Process -Id @($ids) -ErrorAction SilentlyContinue)
    [pscustomobject]@{
        Time = (Get-Date).ToString('o')
        ElapsedSeconds = [math]::Round(((Get-Date)-$start).TotalSeconds,1)
        Processes = $processes.Count
        AggregateWorkingSetMiB = [math]::Round((($processes | Measure-Object -Property WorkingSet64 -Sum).Sum)/1MB,1)
        CurrentTreeCpuSeconds = [math]::Round(($processes | Measure-Object -Property CPU -Sum).Sum,1)
    } | Export-Csv -LiteralPath $Destination -Append -NoTypeInformation -Encoding UTF8
    Start-Sleep -Seconds 30
}


````

### 程序/run_extended.ps1

````powershell
param(
    [Parameter(Mandatory=$true)][string]$OptionFile,
    [Parameter(Mandatory=$true)][string]$SpotFile,
    [int]$Workers = 6,
    [int]$InitialMarketPid = 0,
    [switch]$Publish,
    [switch]$VerifyOnly
)
$ErrorActionPreference = 'Stop'
$directory = Get-Item -LiteralPath $PSScriptRoot
while ($directory -and !(Test-Path -LiteralPath (Join-Path $directory.FullName '.git'))) {
    $directory = $directory.Parent
}
if (!$directory) { throw 'Run from the study repository.' }
$project = $directory.FullName
$work = Join-Path $project '工作区'
$envRoot = Join-Path $work '运行支持/环境'
$out = Join-Path $work '数据/处理后/扩展组合实验'
$marketPython = Join-Path $envRoot 'market/python.exe'
$hedgePython = Join-Path $envRoot 'hedge313/Scripts/python.exe'
$programs = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = "$(Join-Path $programs '创新组合实验');$(Join-Path $programs '完整认购')"
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:TF_CPP_MIN_LOG_LEVEL = '2'
$env:TF_NUM_INTEROP_THREADS = '2'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$lock = [IO.File]::Open((Join-Path $out '.expanded.lock'), 'OpenOrCreate', 'Write', 'None')
try {
    if (!$VerifyOnly) {
        if ($InitialMarketPid -and (Get-Process -Id $InitialMarketPid -ErrorAction SilentlyContinue)) {
            Wait-Process -Id $InitialMarketPid -ErrorAction SilentlyContinue
        }
        & $marketPython -u (Join-Path $PSScriptRoot 'expanded_market.py') --project $project --out $out --options $OptionFile --spot $SpotFile
        if ($LASTEXITCODE) { Write-Warning 'See R1/market_failures.json. Other branches will continue.' }
    }
    $flags = @()
    if ($Publish) { $flags += '--publish' }
    if ($VerifyOnly) { $flags += '--verify-only' }
    & $hedgePython -u (Join-Path $PSScriptRoot 'run_extended.py') --project $project --out $out --workers $Workers @flags
    if ($LASTEXITCODE) { throw "Expanded experiment exited with code $LASTEXITCODE. See the experiment logs." }
} finally {
    $lock.Dispose()
}


````

### 程序/run_extended.py

````python
"""Run, resume, and verify the fixed expanded study."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

from prepare_common import digest, write_json
from extended_data import Inputs
from extended_registry import registry
from extended_replay import baselines, folders, replay, statics
from extended_train import GATES, expert_jobs, state_file, train_job


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sources(project, root, context):
    end, r, rep, market = context
    old = project/"工作区/数据/处理后/创新组合实验"
    fold = old/str(end)
    if r == 0:
        where = fold
    else:
        where = root/"R1"/str(end)
    directory, representation = where/rep, where/f"{rep}.npz"
    marker = directory/"market.done.json"
    manifest = load(marker)
    for name, sha in manifest["outputs"].items():
        if digest(directory/name) != sha:
            raise ValueError(f"Market artifact changed: {directory/name}")
    episode_marker = fold/"episodes.done.json"
    if digest(fold/"episode_days.csv") != load(episode_marker)["outputs"]["episode_days.csv"]:
        raise ValueError("Episode rows changed")
    pool = directory/"pool.npz"
    if market != "single" and not pool.exists():
        if load(directory/"market.json")["members"] < 2:
            return None, None
        raise FileNotFoundError(pool)
    return (fold, directory, representation), dict(market=digest(marker),
                episodes=digest(episode_marker), representation=digest(representation))


def execute(project, root, cfg, token, context, jobs, frozen=None):
    torch.set_num_threads(1)
    phase = "train" if frozen is None else "replay"
    original = project/"工作区/数据/处理后/创新组合实验"
    try:
        files, source = sources(project, root, context)
        if files is None:
            return [dict(**j.meta(), key=j.key, status="unavailable",
                         reason="Fewer than two calibrated market members") for j in jobs]
        inputs = Inputs(*files, next(f for f in cfg["folds"] if f[-1] == context[0]))
        source["source"] = token
    except Exception:
        return [dict(**j.meta(), key=j.key, status="failed", phase=phase,
                     reason=traceback.format_exc()) for j in jobs]
    results = []
    for j in jobs:
        try:
            if frozen is None:
                status = train_job(root, j, cfg, inputs, original, source)
                print(status, j.key, flush=True)
            elif j.key in frozen:
                weights = state_file(root, j)
                record = load(weights.with_suffix(".json"))
                if digest(weights) != frozen[j.key] or record["weights_sha256"] != frozen[j.key]:
                    raise ValueError("Checkpoint changed after freeze")
                replay(root, root/"evaluation", inputs, j, record)
            else:
                continue
        except Exception:
            reason = traceback.format_exc()
            print(phase, "failed", j.key, reason, flush=True)
            results.append(dict(**j.meta(), key=j.key, status="failed", phase=phase, reason=reason))
            if frozen is None:
                write_json(state_file(root, j).with_suffix(".json"), dict(status="failed", reason=reason))
    if frozen is not None:
        for job in jobs:
            if job.policy == "reliable_gate":
                try:
                    if all(x.key in frozen for x in expert_jobs(job).values()):
                        statics(root, root/"evaluation", inputs, job, source)
                except Exception:
                    results.append(dict(**job.meta(), key=job.key, status="baseline_unavailable",
                                        reason=traceback.format_exc(), phase="static"))
        try:
            baselines(root/"evaluation", inputs, context, project/"交付区/结果/创新组合实验")
        except Exception:
            results.append(dict(context=context, status="baseline_unavailable",
                                reason=traceback.format_exc(), phase="baseline"))
    return results


def run_phase(project, root, cfg, token, groups, workers, frozen=None):
    failures = []
    label = "training" if frozen is None else "replay"
    if frozen is None:
        # Bound the size of the largest work unit without reloading for each job.
        chunks = []
        for context, jobs in groups.items():
            for seed in cfg["seeds"]:
                chunks.append((context,[j for j in jobs if j.seed == seed]))
        groups = chunks
    else:
        groups = list(groups.items())
    groups.sort(key=lambda item: -len(item[1]))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(execute, project, root, cfg, token, context, jobs, frozen):
                   (context, jobs) for context,jobs in groups}
        for future in as_completed(pending):
            context,jobs = pending[future]
            try:
                errors = future.result()
            except Exception:
                errors = [dict(**j.meta(), key=j.key, status="failed",
                               phase=label, reason=traceback.format_exc()) for j in jobs]
            failures.extend(errors)
            print(label, context, "complete", len(jobs), Counter(e["status"] for e in errors), flush=True)
    return failures


def freeze(root, jobs, token, failures):
    frozen = {}
    missing = {e["key"] for e in failures if "key" in e}
    for j in jobs:
        if j.key in missing:
            continue
        weights = state_file(root, j)
        status = weights.with_suffix(".json")
        if status.exists() and weights.exists():
            s = load(status)
            if (s.get("status") == "complete" and s["identity"]["source"] == token
                    and s["weights_sha256"] == digest(weights)):
                frozen[j.key] = s["weights_sha256"]
                continue
        failures.append(dict(**j.meta(), key=j.key, status="failed",
                             phase="freeze", reason="Checkpoint missing or stale"))
    return frozen


def archive_replays(root, frozen, include_aux=True):
    """Retain old auxiliary results separately before refreshing this run's set."""
    out = root/"evaluation"
    archive = root/"previous_replays"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    for file in (out/"paths").glob("*.json"):
        payload = load(file)
        registered = payload["meta"].get("registered", False)
        if (registered and file.stem in frozen) or (not registered and not include_aux):
            continue
        for path in (file, out/"daily"/f"{file.stem}.csv.gz"):
            if path.exists():
                target = archive/path.relative_to(out)
                target.parent.mkdir(parents=True, exist_ok=True)
                path.replace(target)


def run(args):
    from extended_report import report
    from extended_delivery import deliver
    from verify_extended import verify
    here = Path(__file__).resolve().parent
    project, root = args.project.resolve(), args.out.resolve()
    (root/"policies").mkdir(parents=True, exist_ok=True)
    out = root/"evaluation"
    folders(out)
    cfg = load(here/"study.json")
    jobs = registry(cfg)
    token = dict(torch=torch.__version__, numpy=np.__version__, files={p.name:digest(p) for p in
        [here/"study.json", *[here/f"{n}.py" for n in
         ("extended_registry","extended_train","extended_data","extended_curve","extended_policy")]]},
        parents={f"{folder}/{n}":digest(here.parent/folder/f"{n}.py")
                 for folder, modules in (("创新组合实验",("train","policy","prepare_common")),
                                         ("完整认购",("contract_ledger","hedge_experiment")))
                 for n in modules})
    write_json(root/"registry.json", dict(source=token, jobs={k:j.meta() for k,j in jobs.items()}))
    verify(project, root, cfg)
    if args.verify_only:
        return
    groups = defaultdict(list)
    for j in jobs.values():
        groups[j.end,j.r,j.rep,j.market].append(j)
    for js in groups.values():
        js.sort(key=lambda j: (j.seed, j.lam, j.policy in GATES|{"capacity"}, j.family != "main",
                               j.w or j.h, j.t != "off"))
    failures = run_phase(project, root, cfg, token, groups, args.workers)
    frozen = freeze(root, jobs.values(), token, failures)
    write_json(out/"frozen_checkpoints.json", frozen)
    write_json(out/"training_failures.json", failures)
    archive_replays(root, frozen)
    replay_failures = run_phase(project, root, cfg, token, groups, args.workers, frozen)
    for e in replay_failures:
        if e["status"] == "failed":
            frozen.pop(e["key"], None)
    write_json(out/"frozen_checkpoints.json", frozen)
    archive_replays(root, frozen, include_aux=False)
    failures = list({(e.get("key"),e["status"],e.get("phase"),str(e.get("context"))):e
                     for e in failures + replay_failures}.values())
    write_json(out/"failures.json", failures)
    completion = report(root, project, cfg, failures, expected=len(jobs))
    if not isinstance(completion, dict):
        completion = load(out/"completion.json")
    if completion["evaluated"] + completion.get("unavailable", 0) == len(jobs) and not completion.get("failures"):
        deliver(root, project, publish=args.publish)
    else:
        raise RuntimeError(f"See failures.json: {completion}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--workers", default=6, type=int)
    p.add_argument("--publish", action="store_true")
    p.add_argument("--verify-only", action="store_true")
    run(p.parse_args())


````

### 程序/study.json

````json
{
  "folds": [
    [
      2022,
      2023,
      2025
    ],
    [
      2019,
      2020,
      2021
    ],
    [
      2020,
      2021,
      2022
    ],
    [
      2021,
      2022,
      2023
    ]
  ],
  "windows": [
    [
      15,
      30
    ],
    [
      45,
      60
    ],
    [
      75,
      90
    ]
  ],
  "moneyness": [
    0.9,
    1.1
  ],
  "report_moneyness": [
    0.97,
    1.03
  ],
  "simulation_train": 32768,
  "simulation_valid": 8192,
  "origin_seed": 104,
  "shock_seed": 105,
  "boundary": "reflect",
  "seeds": [
    0,
    1,
    2
  ],
  "lambdas": [
    0,
    0.1
  ],
  "epochs": 100,
  "validation_every": 10,
  "patience": 3,
  "batch_size": 1024,
  "real_batch_size": 128,
  "bootstrap": 2000,
  "teacher_epochs": 30,
  "warm_epochs": 5,
  "gradient_batches": 8
}

````

### 程序/verify_extended.py

````python
"""Executable checks for curve extensions, alignment, and legacy replay."""
import json
from collections import Counter
import csv

import numpy as np
import torch

from prepare_common import write_json
from contract_ledger import ledger
from hedge_experiment import real_inputs
from extended_policy import DensePolicy, load_legacy, make_policy, rollout, simulated_pnl
from extended_train import predictions
from extended_registry import Job
from extended_curve import bs_delta, curve
from extended_data import Inputs, batch
from extended_registry import registry


def verify_original_interface(project, cfg):
    """Load all original 32-wide models into the new account interface."""
    data = project/"工作区/数据/处理后"
    base, old = data/"创新组合实验"/"2025", data/"完整认购"
    inputs = Inputs(base, base/"C1", base/"C1.npz", cfg["folds"][0])
    paths, observed = inputs.observed(2025)
    state = np.load(old/"joint_paths.npz")
    expected = {}
    with (old/"dh_ledger.csv").open(encoding="utf8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["date"] == row["expiry"]:
                key = (row["group"], int(row["bp"]), int(row["seed"]), row["episode"])
                expected[key] = float(row["wealth"])/float(row["initial_notional"])
    worst, rounding, checked = 0., 0., 0
    for file in (old/"hedge_jobs").glob("*.pt"):
        status = file.with_suffix(".json")
        if not status.exists():
            continue
        meta = json.loads(status.read_text(encoding="utf8"))
        name, bp, seed = meta["group"], meta["bp"], meta["seed"]
        has = name.endswith("factors")
        f, _ = real_inputs(paths, state, has)
        model = DensePolicy("legacy", current=5 if has else 3, width=32)
        load_legacy(model, torch.load(file, weights_only=True))
        job = Job(2025,"C0",0,"off",False,False,bp,0,seed,"single","narrow","baseline")
        pnl, ratios, _ = predictions(model,job,dict(observed,f=f),size=len(paths))
        for p,r,new in zip(paths,ratios,pnl):
            numeraire = float(p[0]["spot"])*int(p[0]["unit"])
            old_pnl = float(ledger(p,r[:len(p)-1],float(p[0]["option_close"]),bp/10000)[-1]["wealth"])/numeraire
            worst = max(worst,abs(old_pnl-expected[name,bp,seed,p[0]["episode"]]))
            rounding = max(rounding,abs(new-old_pnl))
        checked += 1
    state.close()
    assert checked == 18 and worst <= 1e-8, (checked,worst)
    return worst, rounding


def verify(project, root, cfg):
    torch.set_num_threads(1)
    import unittest
    import verify_policy
    assert unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(verify_policy)).wasSuccessful()
    expected = dict(main=1152,narrow=144,teacher=72,history=48,delta=48,bridge=168,cost=144)
    assert Counter(x.family for x in registry(cfg).values()) == expected
    previous = project/"工作区/数据/处理后"
    original = previous/"创新组合实验"
    worst, rounding = verify_original_interface(project, cfg)
    info = {}
    for train, validation, end in cfg["folds"]:
        base = original/str(end)
        inputs = Inputs(base, base/"C1", base/"C1.npz", (train,validation,end))
        _, real = inputs.observed(train, teacher=True)
        assert (~real["teacher_valid"] | real["active"]).all()
        assert inputs.paths[train][-1][-1]["date"] < str(train+1)
        assert len({p[0]["episode"] for ps in inputs.paths.values() for p in ps}) == sum(map(len,inputs.paths.values()))
        sim, _ = inputs.simulation("single")
        initial = np.random.default_rng(cfg["origin_seed"]).integers(len(inputs.paths[train]),size=len(sim["f"]))[:10]
        a = batch(sim, slice(0,10), True)
        b = batch(real, initial, True)
        assert torch.equal(a["history"][:,0],b["history"][:,0])
        assert torch.equal(a["history_valid"][:,0],b["history_valid"][:,0])
        for wealth in (False,True):
            torch.manual_seed(0)
            model = make_policy("wide",wealth,True)
            with torch.no_grad():
                old = rollout(model,a,.0005)[0]
                fast = simulated_pnl(model,a,.0005)
            torch.testing.assert_close(fast,old,atol=2e-6,rtol=0)
        info[str(end)] = dict(paths={str(k):len(v) for k,v in inputs.paths.items()},
                             teacher_valid=int(real["teacher_valid"].sum()),
                             teacher_decisions=int(real["active"].sum()))
        for where in (base,root/"R1"/str(end)):
            repfile = where/"C1.npz"
            if not repfile.exists():
                continue
            with np.load(repfile) as data:
                k = np.linspace(.5,1.5,201)
                for term in (0,1,3,7,14,45,90):
                    t = np.full_like(k,term/365)
                    xi = np.broadcast_to(data["xi"][100],(len(k),2))
                    scale = np.ones(2)
                    c, dk, df = curve(t,k,xi,data,scale)
                    assert np.min(c-np.maximum(1-k,0)) > -1e-8
                    assert np.max(c) <= 1+1e-8
                    assert np.max(np.diff(c)) <= 1e-8
                    assert np.min(np.diff(c,n=2)) >= -1e-8
                    if not term:
                        np.testing.assert_array_equal(c,np.maximum(1-k,0))
                        continue
                    # Test analytic gradients away from the payoff/floor and grid kinks.
                    away = np.min(abs(k[:,None]-np.r_[np.unique(data["grid_k"]),1.]),axis=1) > 1e-5
                    eps = 1e-7
                    cp = curve(t,k+eps,xi,data,scale)[0]
                    cm = curve(t,k-eps,xi,data,scale)[0]
                    np.testing.assert_allclose((cp[away]-cm[away])/(2*eps),dk[away],atol=1e-6)
                    for j in range(2):
                        z = np.eye(2)[j]*eps
                        cp = curve(t,k,xi+z,data,scale)[0]
                        cm = curve(t,k,xi-z,data,scale)[0]
                        np.testing.assert_allclose((cp[away]-cm[away])/(2*eps),df[away,j],atol=1e-6)
    assert list(info["2025"]["paths"].values()) == [2626,387,773]
    delta, valid = bs_delta(np.ones(3), np.ones(3), np.array([.1,.0,1.1]), np.full(3,.1))
    assert valid.tolist() == [True,False,False] and .54 < delta[0] < .56
    result = dict(legacy_original_18_max_error=worst, legacy_account_precision_gap=rounding,
                  registered=1776, folds=info)
    write_json(root/"verification.json", result)
    print("Expanded study verification:",json.dumps(result),flush=True)


````

### 程序/verify_policy.py

````python
"""Independent ledger, gradient and compatibility checks for extended policies."""
import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn

from extended_policy import make_policy, load_legacy, rollout


def batch(spot, unit=None, active=None):
    spot = torch.tensor(spot, dtype=torch.float64)
    n, h = spot.shape
    return dict(spot=spot, unit=torch.full_like(spot, 10.) if unit is None else
                torch.tensor(unit, dtype=torch.float64),
                f=torch.zeros(n, h-1, 8),
                active=torch.ones(n, h-1, dtype=torch.bool) if active is None else
                torch.tensor(active, dtype=torch.bool),
                premium=torch.full((n,), 5., dtype=torch.float64),
                strike_end=torch.full((n,), 100., dtype=torch.float64))


def constant(ratio=.5):
    model = make_policy("wide")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[-2].bias.fill_(torch.logit(torch.tensor(ratio)).item())
    return model


class PolicyChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(1)

    def test_entry_rebalance_and_exit_cost(self):
        data = batch([[100, 110, 90]])
        pnl, ratios, gate = rollout(constant(), data, .001)
        self.assertAlmostEqual(pnl.item(), -.95/1000, places=12)
        torch.testing.assert_close(ratios, torch.full_like(ratios, .5))
        self.assertIsNone(gate)

    def test_dividend_entitlement_adjustment_and_expiry_payment(self):
        data = batch([[100, 98, 104, 102]], [[10, 11, 11, 11]])
        data.update(strike_end=torch.tensor([99.], dtype=torch.float64),
                    record_event=torch.tensor([[7, 0, 0, 0]]),
                    ex_event=torch.tensor([[0, 7, 0, 0]]),
                    pay_event=torch.tensor([[0, 0, 0, 7]]),
                    record_amount=torch.tensor([[2., 0, 0, 0]], dtype=torch.float64))
        pnl, _, _ = rollout(constant(), data, .001)
        self.assertAlmostEqual(pnl.item(), 37.89/1000, places=12)
        data["pay_event"].zero_()
        unpaid, _, _ = rollout(constant(), data, .001)
        torch.testing.assert_close(unpaid, pnl, rtol=0, atol=1e-14)

    def test_padding_settles_once_and_does_not_recharge(self):
        data = batch([[100, 110, 110, 110], [100, 110, 90, 100]],
                     active=[[True, False, False], [True, True, True]])
        pnl, ratios, _ = rollout(constant(), data, .001)
        self.assertAlmostEqual(pnl[0].item(), -1.05/1000, places=12)
        self.assertEqual(ratios[0, 1:].abs().sum().item(), 0)

    def test_student_wealth_and_position_keep_cross_time_gradient(self):
        class Probe(nn.Module):
            def __init__(self):
                super().__init__()
                self.theta = nn.Parameter(torch.tensor(-.4, dtype=torch.float64))
            def prepare(self, data):
                return data["f"].transpose(0, 1)[..., 0]
            def action(self, prepared, previous, wealth):
                return torch.sigmoid(self.theta + 10*wealth + .2*previous), None
        data = batch([[100, 110, 100]])
        model = Probe()
        actual, _, _ = rollout(model, data, 0)
        gradient = torch.autograd.grad(actual.sum(), model.theta)[0]
        a = torch.sigmoid(model.theta + .5)
        b = torch.sigmoid(model.theta + 10*(.05 + .1*a) + .2*a)
        expected = .05 + .1*a - .1*b
        reference = torch.autograd.grad(expected, model.theta)[0]
        torch.testing.assert_close(actual[0], expected, rtol=0, atol=1e-13)
        torch.testing.assert_close(gradient, reference, rtol=0, atol=1e-13)
        center = model.theta.item()
        with torch.no_grad():
            model.theta.fill_(center + 1e-5)
            plus = rollout(model, data, 0)[0].item()
            model.theta.fill_(center - 1e-5)
            minus = rollout(model, data, 0)[0].item()
        self.assertAlmostEqual(gradient.item(), (plus-minus)/2e-5, places=9)

    def test_static_gate_endpoints_share_the_actual_account(self):
        data = batch([[100, 110, 90]], [[10, 11, 11]])
        data["f"].normal_()
        experts = dict(base=make_policy("base", wealth=True),
                       factor=make_policy("wide", wealth=True))
        for weight, name in [(0., "base"), (1., "factor")]:
            mixed = make_policy("static", wealth=True, experts=experts, fixed=weight)
            actual, _, gate = rollout(mixed, data, .0005, capture=True)
            expected = rollout(experts[name], data, .0005)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
            torch.testing.assert_close(gate, torch.full_like(gate, weight))
        gate = make_policy("reliable_gate", wealth=True, experts=experts)
        rollout(gate, data, .0005)[0].square().sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0
                            for p in gate.gate.parameters()))
        self.assertTrue(all(p.grad is None for p in gate.base.parameters()))

    def test_history_models_are_causal_and_capacity_matched(self):
        data = batch([[100, 110, 90, 105]])
        data["history"] = torch.randn(1, 3, 10, 4)
        data["history_valid"] = torch.ones(1, 3, 10, 4, dtype=torch.bool)
        target = sum(p.numel() for p in make_policy("wide", wealth=True, history=True).parameters())
        for kind in ["wide", "flat", "gru"]:
            model = make_policy(kind, wealth=True, history=True)
            size = sum(p.numel() for p in model.parameters())
            self.assertLessEqual(abs(size-target)/target, .05)
            original = rollout(model, data, .0005)[1]
            changed = dict(data, history=data["history"].clone())
            changed["history"][:, 2] += 100
            updated = rollout(model, changed, .0005)[1]
            torch.testing.assert_close(original[:, :2], updated[:, :2], rtol=0, atol=0)

    def test_main_networks_keep_fixed_slots_when_features_are_disabled(self):
        for kind in ["wide", "narrow", "base"]:
            shapes = []
            for wealth, history in [(False, False), (True, False), (False, True), (True, True)]:
                shapes.append(tuple((key, tuple(value.shape)) for key, value in
                                    make_policy(kind, wealth=wealth, history=history).state_dict().items()))
            self.assertTrue(all(shape == shapes[0] for shape in shapes))

    def test_residual_zero_head_reproduces_clipped_teacher(self):
        data = batch([[100, 101, 102, 103]])
        data["delta"] = torch.tensor([[0., .3, 1.]])
        data["delta_valid"] = torch.ones(1, 3, dtype=torch.bool)
        _, ratios, _ = rollout(make_policy("delta_residual"), data, 0)
        torch.testing.assert_close(ratios, data["delta"].clamp(1e-4, 1-1e-4), rtol=0, atol=1e-7)

    def test_old_weight_mapping_and_decisions(self):
        file = Path(__file__).parents[1]/"创新组合实验"/"policy.py"
        spec = importlib.util.spec_from_file_location("legacy_policy_check", file)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        data = batch([[100, 101, 102, 103], [100, 95, 98, 100]],
                     [[10, 11, 11, 11], [10, 10, 10, 10]])
        data["f"].normal_()
        for kind, width in [("wide", 46), ("narrow", 32)]:
            original = old.MLP(8, width)
            model = make_policy(kind)
            load_legacy(model, original.state_dict())
            expected = old.rollout(original, data["f"], data["unit"][:, :-1].float())[0]
            actual = rollout(model, data, .0005)[1]
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)

````
