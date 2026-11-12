# 冻结因子表征的独立 masked-quote 审计

新增 factor_quote_audit.py、test_factor_quote_audit.py 和本报告，没有改动basis、center、W、bounds、行情或既有模型，也未运行真实全量审计。

## 命令、正式输出和状态

`工作区/运行支持/环境/volatility/Scripts/python.exe factor_quote_audit.py --end 2025 --representation nsde`。

representation支持nsde/svi/ssvi/essvi/original。nsde读取该折旧C1.npz；三种新表征读取 `OUT/representations/end/{svi,ssvi,essvi}_linear.npz`。original是原未压缩20价格网格的静态参照，不是第10种生成器。

正式输出 `OUT/factor_quote_audit/end/representation/`：

- identity.json：source与所有parents SHA、折分割、冻结种子、仅训练期五初值及其SHA。
- daily.jsonl：逐date/view的xi或original grid、优化尝试/修复、金融slack、价格指标、样本数、耗时、out-of-grid及identity SHA。
- daily.csv：上述可汇总指标扁平表。
- summary.json：training/validation/historical_evaluation × full/heldout的报价数加权RMSE、天数、失败数、最小金融slack、超网格比例。
- status.json：只有所有fit合格才complete=true/status=complete；正常优化诊断失败为fit_failed；missing basis或异常为failed并重新抛错。输出文件均有SHA。没有存在即缓存复用规则。

日JSONL/CSV先临时写后replace；开始标running，最终状态最后写。仅使用date<end+1的数据；在报价清洗前先截断截止日期，避免未来异常报价影响当前审计。

## 数据掩码和冻结初值

full视图用全部当日报价拟合，仅用于描述性曲面重构；heldout视图重新独立拟合，只使用 `~holdout` 报价。每个视图的visible和heldout统计永远采用互斥掩码，分别记录n、SSE和RMSE。full的heldout误差属于样本内重构，明确 `heldout_is_out_of_sample=false`；只有heldout视图为true。两个视图不得混在一起汇总为一个留出误差。

按(date,expire_date,effective strike)检查重复；测试无这些字段时按(date,ttm,k)检查。相同有效报价重复且目标/掩码一致时去重；同一有效报价落入不同掩码或目标冲突立即拒绝，防止重复K跨mask泄漏或重复加权。

五初值从rep.dates<训练年末+1的rep.xi固定抽取，seed=20261112；训练池少于5条时允许固定抽样重复，仍固定五次优化。绝不采用当日full xi、前一个评价日xi、masked目标或holdout报价来选初值。训练池/初值SHA在full和heldout之间完全相同。

## 两因子拟合

预测和解析因子梯度均复用既有 `extended_curve.curve(tau,k,重复xi,rep,scale=np.ones(2))`，不重新定义查询插值。优化SLSQP的等权归一化call价格MSE，约束 `W@xi>=bounds`，jac直接使用该curve返回的df，约束jac为冻结W。内部目标乘10000仅用于数值尺度，输出objective仍为原MSE。

每个初值最多300步、固定ftol=1e-12，未根据评价表现选参数。先在成功且slack≥−1e-7的候选中按MSE取最优；若无合格候选，保留失败诊断并使该次审计不complete。原NSDE也强制以当日visible重新拟合masked state，没有把原full xi直接拿来评价heldout。

原curve的0期限payoff锚、行权价左尾零执行价锚、右尾受限斜率、内在价值floor、价格cap及期限边界策略全部沿用；记录报价落在原14/30/60/90天×.9..1.1网格以外的比例。该金融slack只保证冻结有限网格约束，不宣称连续全域无套利。

## Original未压缩网格参照

每个date/view先只用该视图允许报价，按原market_surfaces构建价格网格：每到期按执行价插值，K=0价格1为锚；右端原实现使用固定−.5斜率（属于[-1,0]内）并以内在价值为下界；期限方向加入T=0 payoff，再在原网格节点插值，最长到期外采用原np.interp常值边界。

约束只读取既有surfaces.npz的A、b、grid_t、grid_k，不读取其full calls作为masked目标。若raw grid不满足原A@raw>=b，调用scipy linprog实现原L1修复：生产20网格对应40个非负u/v，目标sum(u+v)，`Aub=[-A,A]`、`bub=A@raw-b`，修复grid=raw+u−v。检查最终slack≥−1e-7，记录修复L1、均值和raw/final slack。没有cvxopt依赖。

查询时以修复grid作center、零二维basis/xi复用同一extended_curve；没有重新拟合两个factor。original各视图完全独立从对应报价重建；改变heldout target不能改变masked grid。该参照保留更多价格自由度，不能把其结果当成参数数量相同的因子模型。

## 分割和统计

training为date<fold[0]+1，validation为训练截止后且date<fold[1]+1，其余截至end属于historical_evaluation；明确记录日期范围规则，不根据路径盈亏选日期。summary使用各日SSE总和/报价n总和开方，不用不等样本数的日RMSE简单平均。full和heldout严格分开汇总。identity记录完整父文件SHA，daily记录同一identity SHA。

## 测试证据（2026-11-12）

先写3项测试，在模块不存在时全部明确断言失败。另增加截止日期回归，先重现未来2026年坏报价导致2025年审计失败，再修复为先按end截断后清洗。

最终命令：`工作区/运行支持/环境/volatility/Scripts/python.exe -m unittest discover -s 工作区/当前研究/程序/波动率模型比较 -p test_factor_quote_audit.py -v`。

4项通过，约3.5秒：

1. BS近似两因子可行basis生成报价，masked xi恢复[.01,−.005]；把所有hidden目标改成.8、当前full xi改成[−999,−999]，masked xi逐位不变，只有heldout误差变化。
2. visible=6、heldout=4，full fit_n=10、masked fit_n=6；同一有效K跨mask重复直接拒绝。
3. 原grid负价的linprog L1修复确实满足A,b；masked原grid在隐藏目标极端改变后逐位不变。
4. 隔离临时两日期审计真实输出training/historical_evaluation两组、排除截止后的异常报价；缺新svi basis时明确failed receipt且抛FileNotFoundError，未静默使用nsde basis。

这些是局部正确性和抗泄漏证据，不是全量静态比较结果。尚未运行每折所有日期，未据此宣称原NSDE或任何压缩模型的实际留出效果。
