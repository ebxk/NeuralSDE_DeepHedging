# 波动率曲面模块实施与验证

## 接口和使用边界

`volatility_surfaces.py` 不读写行情文件，仅接受 numpy 可转换数组。`fit_surface(kind, ttm, k, prices, seed=20261112, starts=5)` 中 ttm 为年，k 为 K/spot，prices 为 call/spot；返回能通过 `json.dumps(..., allow_nan=False)` 的字典。kind 为 `svi`、`ssvi`、`essvi`。`price_surface(model,t,k)` 支持广播，t≤0 返回内在价值。`summary(model)` 返回 30/365 年 ATM IV、IV(1.05)−IV(.95)，以及 atm_valid/skew_valid；无效数值为 None。

模型采用 r=q=0、spot=forward=1 的统一口径。实际非零利率和股息行情必须由调用方先转成一致的远期/贴现归一化口径，不能直接把本接口解释为含真实融资成本的定价器。三模型共同优化全部报价等权的归一化价格 MSE，没有用 IV 误差替代价格误差。内部目标乘 10000 改善数值尺度，返回的 objective 仍是原 MSE。

## 参数、插值、约束

- raw SVI：每期限 a,b,rho,m,sigma，标准 raw 总方差公式。边界 a∈[-2,4]、b∈[0,2]、rho∈[-.98,.98]、m∈[-1,1]、sigma∈[.005,2]；检查最小总方差和尾部斜率界。优化阶段在 log-moneyness [-1,1] 的 61 点检查 Durrleman g 和相邻总方差差。跨期按同一 log-moneyness 的总方差线性插值；短期总方差按 t/T1 缩放，末期限后保持总方差不变。这不是 raw SVI 连续全域无套利保证。
- SSVI：期限 theta，加共用 rho、eta；phi(theta)=eta/sqrt(theta(1+theta))，即 power-law gamma=.5。theta∈[1e-8,4] 且不下降、rho∈[-.98,.98]、eta∈[0,2]。约束 eta²(1+|rho|)≤4 和 eta(1+|rho|)≤4。theta 按期限线性插值，短期趋零，长期按末期限 theta/T 斜率外推。
- eSSVI：每期限 theta,rho,psi。theta∈[1e-8,4]、rho∈[-.98,.98]、psi∈[1e-8,2]。检查 theta 和 psi 不下降、|Δ(rho psi)|≤Δpsi、psi(1+|rho|)≤4、psi²(1+|rho|)≤4theta。插值的是 theta、psi、rho*psi，而非直接插值 rho。短期 theta、psi 同比例趋零且 rho 固定；长期 psi、rho 固定而 theta 按末期限 theta/T 增长。

eSSVI 条件、插值和外推来自 Corbetta, Cohort, Laachir, Martini (2019), *Robust calibration and arbitrage-free interpolation of SSVI slices*, [arXiv:1804.04924](https://arxiv.org/html/1804.04924)，尤其第 2、5 节。实现使用同时约束的 SLSQP 价格 MSE 优化，并未声称复现原文 anchored/逐期限 Brent 校准算法。金融参数界限是本次实验预设的可行域，不是对任意市场的普适结论。

五个初值由固定 seed 产生，调用 starts≠5 会拒绝，避免模型之间改变重启预算。第一个初值以样本中可识别 IV 的中位数确定平坦期限结构；其余初值用同一随机种子扰动。没有按样本外表现调参。SciPy SLSQP 为局部优化，不保证全局最优。

## 诊断

diagnostics 包含 valid、constraint_min、objective、seconds、optimizer_success、seed、starts 和每次优化尝试。valid 要求所选优化成功、约束容差达标和事后网格检查均通过。

事后检查至少约 85×241 点（实际期限去重后决定）：期限包含零、全部校准期限、从 T1/10 到 1.25*Tmax 的 81 点；k 至少覆盖 [.5,1.5]，并随样本执行价外扩。真正检查 call 内在价值下界、上界 1、执行价单调性、斜率≥−1、斜率差凸性以及期限单调性；检查结果和网格边界保存在 diagnostics.grid。容差为 1e-7，极端尾部差分可能出现约 1e-14 的舍入负值。

统一标记 `scope=finite_grid_only`、`global_arbitrage_certified=False`。即使 SSVI/eSSVI 参数满足文献充分条件，当前报告仍仅把程序结果作为有限精度和有限网格诊断；不将网格通过冒充全域证明。失败返回 diagnostics.valid=False，不回退到其他模型冒充成功。

## 测试证据（2026-11-12）

首先创建测试，在实现文件不存在时用 hedge313 Python 运行，4 项均因“surface module must exist”断言失败。随后实现并使用已有 market 环境（numpy 1.22.4，scipy 1.13.1）验证；没有安装依赖或改变环境。

最终命令：`工作区/运行支持/环境/market/python.exe 工作区/当前研究/程序/波动率模型比较/test_volatility_surfaces.py`。

结果：6 项测试通过，0.941 秒。覆盖 BS/IV 往返、广播与到期价格、非法报价和期限、三模型平坦合成校准及 JSON 序列化、eSSVI 的 rho*psi 插值、网格识别明确的日历与蝶式套利、非平坦 eSSVI 拟合和相同种子重复性。未运行真实数据全量任务，也未运行项目全量实验。

补充 3 期限、33 报价的非平坦 eSSVI 合成样本共同拟合：

| 模型 | 价格 MSE | constraint_min | 秒 | 网格 valid |
|---|---:|---:|---:|---|
| raw SVI | 3.1248e-12 | .0025575 | 5.191 | true |
| SSVI | 5.1582e-7 | .0079591 | .052 | true |
| eSSVI | 3.9157e-16 | .0080000 | .118 | true |

这些只是合成正确性检查，不是实证模型优劣结论。SciPy 测试运行会发出“Values in x were outside bounds during a minimize step, clipping to bounds” RuntimeWarning；这是 SLSQP 试探步裁剪，未屏蔽警告，最终约束与网格均独立核验。

## 局限

IV 反解上限固定为年化 10；超过上限或不可识别报价返回 NaN，summary 转为 None 并将标志置 false。拟合要求正期限、正执行价、有限数值及 call 价格范围；没有新增真实市场报价清洗、利率拟合、bid/ask 加权、缺失期限识别或样本外评估逻辑。极少数报价也可数值拟合，但不意味着参数已统计识别。只有优化器成功且诊断通过才可供调用方继续研究；未经真实数据阶段验证不得进入交付区。


## 2026-11-12 候选选择器修复与追溯

在root确认停止旧曲面/训练进程并snapshot旧kernel后，仅修改本模块及其测试。旧选择器按(not feasible,loss)排序，忽略优化器成功与逐候选真实网格；可能让更低MSE的失败候选遮住已成功的合格初值。

现对全部五候选实际计算金融margin和稠密call网格，qualified要求optimizer success、有限objective、financial margin≥−1e-7、grid valid同时满足。只从全部qualified候选中选最低价格MSE；没有qualified才返回最低失败MSE用于诊断，valid保持false。每项attempt记录qualified/grid_valid/constraint_min/grid_margins/optimizer_status/iterations/message。顶层记录qualified_candidates、从0计数的selected_start与selection_rule。未放宽任何金融/网格容差，没有按评价表现更改初值或目标。

新增4项回归：成功候选不得被更低loss失败optimizer挤掉；参数margin容差内但实际call calendar网格失败的候选不得挤掉有效候选（仅模拟optimizer，真实价格/约束/grid计算保持实际）；全部失败明确invalid；真实2015-09-14 eSSVI全93报价反例。先运行出现3项失败、1项缺少新诊断字段，再修复后10项测试全通过，约1.935秒。

真实反例：旧start0 optimizer iteration limit，loss=2.2782014379e-6、grid valid，但success=false，旧模块因此错判整日不合格。其他4个start成功合格。新选择0-based start4，loss=2.3971147162e-5、financial min=7.4797e-14、grid valid，整体valid=true。选择的是合格候选最低loss，而非把失败候选标为成功。真实fixture由本地只读quotes.csv读取；无数据的独立环境会显式skip该项，其余选择器回归仍独立可运行。

没有修改任何旧结果、执行迁移、重新拟合全量或晋级交付区；旧有效结果和失败结果的迁移由root另行保留追溯处理。
