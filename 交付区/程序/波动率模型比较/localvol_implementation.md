# 冻结起点 eSSVI 的局部波动率实现

仅新增 local_volatility.py、test_local_volatility.py 和本报告。没有读取原始行情、没有改动既有模块或环境、没有执行真实训练或全量实验。

## 接口与坐标

- `build_table(surface,horizon_days=120)`：接受起点 eSSVI 字典，返回包含 numpy 的表，字段为 times、spot_axis、local_variance、surface_hash、table_hash、diagnostics。默认 121 个日度时间点、401 个等距 log-spot 点，spot/原点spot 范围 [.2,3.5]。时间单位年。表不是 JSON 成品，应由调用层自行选择持久化方式。
- `simulate_p(table,start_spot,times,shocks,spot_drift=0.)`：times 为包含初始点的绝对起点时间，shocks 为 `[...,steps,2]` 独立标准正态数；仅第零列用于一因子局部波动率，第二列明确未用。返回 spot 路径及 moments/boundary diagnostics。start_spot 单位为原点spot。
- `conditional_prices(table,time,spot_ratio,tenor,strike_ratios)`：time 是从起点经过的年数，tenor 是剩余年限，spot_ratio=S_t/S_origin，strike_ratios=K/S_origin；返回 prices=C_t/S_origin、delta=dC/dS、spot_grid、price_grid、delta_grid、cache_key、surface_hash、diagnostics。支持多个执行价共用一次 PDE 多右端求解。

关键限制：条件价格使用起点冻结表的绝对时钟系数 sigma(time+u,S/S_origin)，不把当期 spot 错当作新的起点，不读取以后日期的任何曲线。time+tenor 必须在表的 120 天覆盖内。120 天不是“每个条件日之后再向前 120 天”；更长合约需事前建立相应 horizon。

## Dupire 计算和非静默数值处理

基础公式为 `sigma_local²=2 C_t/(K² C_KK)`，r=q=0。程序同时在期权价格上计算中心差分 Ct、Ckk，并用相同曲面总方差 w 对 log-strike x 的有限差分计算数学等价形式 `w_t/g`，其中：

`g=(1-x*w_x/(2*w))²-(w_x²/4)*(1/w+1/4)+w_xx/2`。

参考总方差密度 g 的定义：[Gatheral & Jacquier, Arbitrage-free SVI volatility surfaces](https://www.ma.imperial.ac.uk/~ajacquie/index_files/GJ%20-%20NoArbSVI.pdf)。

价格直接差分分母 K²Ckk>1e-9、候选方差为正且与稳定形式相对误差≤1e-5 时使用直接分支。其他点使用等价总方差稳定分支，避免短期深尾价格相消导致 0/0。这一稳定分支方案已获本任务明确接受，属于等价 Dupire 计算，不是事后调平价格或修补负方差。

诊断完整记录 tiny_price_denominator_points、negative_price_candidate_points、price_branch_fraction、stable_branch_fraction、方差范围、g_min；clipped_points 固定为零。w≤0、w_t≤0、g≤1e-8 或非有限/非正最终局部方差立即报错，绝不用上下限替换为正数。价格分支的数值负候选可在稳定分支验证通过时弃用并计数；总方差稳定分支真负值始终拒绝。

差分 h_x=.001、价格 h_K=.0002K、h_t=min(1e-5,t/4) 年。t=0 以 1e-5 年的短期极限近似。eSSVI 期限节点的时间导数采用两侧差分平均，其斜率不光滑会成为离散误差来源，未声明精确连续 Dupire 解。

## 条件 PDE 与缓存

在 x=log(S/S_origin) 网格求解反向方程：`V_t+.5 sigma²(t,S)(V_xx−V_x)=0`。使用最多半日的时间步、Crank–Nicolson，前两步全隐式压制 payoff 折点振荡。三对角求解支持多执行价矩阵右端。边界固定为有限端点的内在价值，零利率条件下与远尾渐近行为一致；这是有限域近似，不是无限域边界证明。

空间结果用 Cubic Hermite 插值，Delta 为同一价格插值函数的精确一阶导数，避免线性价格插值和独立平滑 Delta 彼此不一致。到期时直接返回精确内在价值，Delta 在 ATM 约定 .5。

surface_hash 仅由定价相关曲面参数确定，不受 diagnostics.seconds 等运行噪声影响。table_hash 进一步包含网格与完整系数数据。PDE 缓存键包括 table_hash、time、tenor 和全部执行价，不包括当期 spot，因此同一条件日/到期/合约组的不同路径可共用整个价格网格。缓存返回副本，调用方修改返回价格不会污染缓存。不同源曲面 hash 不可共用。调用方应把表视为冻结对象，不应原地修改 table 数组或哈希字段。缓存为进程内缓存，当前未做磁盘落地或长期容量管理。

## P 模拟和边界诊断

模拟用对数 Euler：`S_next=S exp((mu−sigma²/2)dt+sigma sqrt(dt)Z)`，Q/P 共同使用起点冻结扩散，只改变现货漂移。不裁剪 spot。若 spot 超出 [.2,3.5]，仅为继续生成可审计路径而采用边界局部方差延拓，同时计数并将 diagnostics.valid=False；因此该结果不能被下游静默当作通过训练诊断。条件 PDE 若 spot 超域则直接报错。

moments diagnostics 返回终值均值、给定漂移对应理论均值、均值误差、终值方差、边界计数和 source hash。其为模拟抽样检查，不是全体市场/时间步的鞅证明。

## 测试证据（2026-11-12）

先写 5 项测试，在模块不存在时全部因明确存在断言失败。实现后，Delta 与价格差分的不一致测试先失败，再改为同一 Hermite 插值导数后通过。增加到期测试，重现网格插值导致到期价格不等于内在价值，随后修复并通过。

最终命令：`工作区/运行支持/环境/market/python.exe -m unittest discover -s 工作区/当前研究/程序/波动率模型比较 -p test_local_volatility.py -v`。

结果：6 项通过，约 .701 秒。覆盖常波动 Dupire 表、无静默 clip、BS 条件价格误差<1e-3、解析 Delta 与价格有限差分、同 hash 缓存稳定/不同 hash 隔离、负日历总方差拒绝、20,000 条共享 shock 路径矩、无未来曲线接口和 horizon 拒绝、精确到期价格、非平坦 eSSVI 起点曲线回收。

具体数值：BS sigma=.2、time=.04、条件 spot=1.05、tenor=.15、K/S_origin=[.9,1,1.1]，最大条件价格绝对误差 **1.6429e-5**，最大 Delta 误差 **.0003130**。局部方差范围 [.0399996001,.0400003998]，clipped_points=0；稳定分支占 76.4535%，直接价格分支占 23.5465%，深尾极小分母点 27,757，价格数值负候选 51 点均有记录。

非平坦 eSSVI 三期限 [.05,.15,.4]、参数 [theta,rho,psi]=[.004,−.6,.05][.012,−.4,.08][.032,−.3,.12]：无 clip，直接分支 64.8297%、稳定分支 35.1703%。time=0 从 PDE 回收起点曲线，在到期 .05/.15/.3、执行价 [.9,1,1.1] 的最大误差分别 **3.22e-5 / 1.65e-5 / 8.79e-6**。这些是小规模数值正确性证据，不是实证训练完成或模型绩效结论。

## 研究使用前仍需检查

本次只固定一组离散网格和差分步长，未完成真实参数域上的时间/空间/差分收敛研究。有限边界、eSSVI期限节点导数、极端尾部、很短剩余期限及高局部方差均会影响 PDE 精度；例如合成非平坦曲面在远尾的最大局部方差约 5.03，并未裁剪。真实研究必须保存这些诊断、检查路径出界率、对代表性参数做收敛复核，再决定是否进入交付区。
