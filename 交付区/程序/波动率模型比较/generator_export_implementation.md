# 共同生成器输入导出实施

新增 generator_export.py、test_generator_export.py 和本报告。不修改 common_inputs、feature_scale 或任何评价结果，不启动真实全预算生成/训练。

## 命令和完成信号

`python generator_export.py --end 2025 --model sabr`，model 支持 original_linear/svi_linear/ssvi_linear/essvi_linear/heston/sabr/localvol。公开接口 `export_generator(end,model)`。

默认路径数严格取 CONFIG 的 32768+8192=40960，并要求旧折配置和共同 common.npz 预算相同；没有缩减预算开关。测试用明确隔离的临时目录和 CONFIG=4+2 fixture，不将测试输出冒充真实研究完成。

输出到 `OUT/inputs/end/sim_model.npz`，先写同目录 `.npz.tmp`，关闭文件后原子 replace。最后才写 `sim_model.ready.json`，包含 complete=true、input_sha256、source、parents、identity、诊断、运行时间。队列只能在 ready 存在且 SHA 对应时消费。开始重新导出时撤下旧 ready；失败写 `sim_model.failure.json` 并重新抛出异常，绝不吞错或发布 ready。若失败发生于原子 npz replace 后，npz 可存在，但缺 ready，因此仍不得消费。

## 共同信息和八槽位

读取旧折 C1/common.npz、C1.npz、config.json、episode_days.csv。按原 market.prepare_origins 的 pandas 数字 episode groupby(sort=True)，过滤原训练 split，使用 origin_rng=104 一次抽取全部起点。逐行核验日期相对时间、strike、premium 和共同 GBM 缓存初始 spot。

沿用 common 的 time/warm/strike/premium。warm 的20个历史收益先于模拟收益拼接，滚动实现与 common_inputs 完全相同；第一时点 RV 仅由共同 warm 决定。输出字段与 sim_gbm.npz 相同：

- f: `(N,H-1,7)`，base3=[20log(S/K),12tau,4RV]，summary4=[30d ATM IV,30d skew,两个有效标志]；策略另加前仓，合计八槽位。
- spot/unit: `(N,H)`，unit 全1；active: `(N,H-1)`，只看共同时间是否推进。
- premium、strike_end: `(N,)`。

不自行拟合或修改 feature_scale。所有 IV 摘要通过 linear_market.shared_summary 同一个离散价格算子提取。

## 生成器分支

四个 linear 分支要求 market.json.status=complete，直接读取其 simulation.npz 的 spot/summary，并严格核验 time/warm/strike/premium、行数/起点；重新构建共同训练 features，不读取策略表现。

Heston/SABR 要求 global.json、p_parameters.json、states.csv 和 complete status。按抽中的真实训练日期映射因果 origin state。Q 参数和扩散冻结；P 只改变估计漂移。SABR 优先使用显式 state_log_drift，兼容旧式 state_drift 的算术单位。

共享 shock_rng=105，先生成 `(N,H-1,4,3)` 的三维四子步正态数，再每相邻两步求和除 sqrt(2) 得两子步；按连续512行分块取随机数，不改变随机流顺序。spot 用第0维、随机波动因子用第1维并经 Q rho 相关变换，第2维保留给共同原始冲击结构。每第8条路径用相同四子步重新生成配对细网格诊断，记录 spot 终值 log RMSE、state 终值 RMSE。Heston 记录负候选方差截断次数/有效更新数/比例。恰为0的 CIR 状态做 Q 定价时使用 v=1e-12 连续边界近似，并单独记录数量；不把它隐藏为原始正状态。

摘要只模拟 shared_summary 实际需要的14/30天两行×5执行价，共10点，省略永远不参与30d线性算子的60/90天行；固定最多16个模拟状态另算原完整20点，比较摘要最大误差，验证数学等价。未用 BS 近似替代 Heston/SABR。Heston另记录自适应积分回退率。

## Local volatility 分支

完全独立于 SABR 产物。其现货 P 算术漂移来自该折合法训练现货收益：`sum(r)sum(ACT365dt)+.5*mean((r-logmu*dt)^2/dt)`。只用 rep 训练日期、calendar 相邻索引、排除除息和缺失日期，并保存样本数、最后训练日、训练截止、公式和方差。

每个抽中的 origin date 优先读取当天 essvi/full.json，要求 status complete 且模型数值诊断 valid；当天不合格或不存在时，只向过去交易日查找最近合格曲面，最多5个交易日。仍无合格曲面则由共同warm里该日起过去20日收益的RV构造严格flat起点（theta=RV²T、rho=0、psi=0）。每个origin记录fallback_source、source_date、stale_days和RV；不会读取未来合格曲面。选中合格曲面后再建立120天冻结局部方差表，其真实Dupire负方差/PDE失败绝不触发此可用性fallback，而是numerical_failed。表源 hash 严格区分，不读取未来曲线。用同一两/四子步 shock 生成路径；出局部系数网格则失败，不静默接受边界修补。

每个 origin-date/elapsed-time/14或30天 tenor 共用一次多执行价 Q PDE 网格，以固定201个原点执行价节点解决不同路径的当前 K/S 问题；再对 S 和 K 两维插值并除当前 spot，成为当期归一化价格。每组固定首个模拟状态另按真实执行价直接 PDE 求解，比较归一化价格和 IV。要求 price误差≤1e-4、IV误差≤.005；否则仅允许一次201→401执行价节点加密，仍失败则抛出 numerical_failed，无 ready。每个 origin 完成后清理进程 PDE 网格缓存，避免跨数百起点积累内存；source hash 仍保持严格区分。

120天范围只覆盖路径约90天加摘要30天；没有生成无用60/90天条件节点，也没有扩大到180天或伪造其价格。

## 测试和成本证据

先写3项测试，在模块不存在时全部明确断言失败。实现后加真实 linear/LV 端到端临时目录测试。最终5项测试通过，约6.7秒，覆盖：

1. 六条真实 SABR 数值路径导出，shared warm 和起点、当前state Q摘要、7features+前仓=8、atomic ready/SHA、无遗留 tmp。
2. shock_rng105 四→二聚合与单次全数组一致、Heston负方差候选截断计数。
3. 缺少校准产物必须抛错、落失败receipt、不发布ready。
4. 完成linear缓存导出以及常波动eSSVI→真LV PDE→输入导出，验证归一化价格/IV插值门限。
5. 起点曲面只能用过去≤5交易日状态；过去状态撤下后使用RV20 flat，明确忽略已经存在的未来合格曲面。

命令：`工作区/运行支持/环境/market/python.exe -m unittest discover -s 工作区/当前研究/程序/波动率模型比较 -p test_generator_export.py -v`。

仅只读规划计数了2025折 origin，未生成其真实路径：40960抽样覆盖2626条训练 episode、453个起点日期、16400个 origin-clock 组；两tenor至少32800次共享PDE网格求解，另有固定查询审计/可能一次加密。隔离六路径、两个起点四时刻样本包含16组PDE，约5–6秒；粗略外推约2–4小时/折，存在曲面/机器负载差异，不是已完成运行或保证时限。Heston大量条件价格亦有显著 CF 成本，分块只限制内存，不减少样本量。

## 局限

没有运行真实全预算导出，因此不能宣称任一新随机生成器的真实40960路径已完成。LV固定审计只覆盖每个条件组一个预定状态，并非每条路径；尾部误差仍需结合训练数值诊断解释。若真实曲面建表不合法、PDE门限失败、因果状态缺失或全局诊断失败，导出明确失败，不用其他模型回填。代码未读取策略收益、测试损失、模型排名等评价表现。


## 追加：条件诊断状态sidecar

每模型另atomic写sim_model.state_sidecar.npz；ready.state_sidecar给file/sha256/complete。通用origin_dates、episode_id、selected_origin_index；Heston/SABR含latent(N,H)，线性含原xi，LV含origin_surface_hash。它们不进入f，也不改变预算/主训练数组。旧ready无此字段或缺少匹配SHA的sidecar时，不能宣称1/5/20条件诊断已完成，必须重新真实导出。回归已先失败后通过，验证latent初态、episode顺序和SHA。
