# 静态直接对冲费用/执行矩阵修复（I6）

仅修改direct_hedges.py，新增test_direct_static.py及本报告。未修改旧财务账本、曲面拟合器、数据、权重或其他源码，也未运行全量真实评价。

## 场景和输出

BS Black，以及SVI/SSVI/eSSVI各自Black/冻结参数Smile Delta，共7种定义；每种定义均评价0/5/10bp同收盘与5bp下一收盘，合计28个场景组。

沿用既有命令 `direct_hedges.py --end 2025`，使用含torch的volatility环境。沿用 `OUT/direct/end/path_pnl.csv`、`daily_ledger.csv`、`ratios.csv`，新增 `group_metrics.csv`；各表键包含end/model/delta/cost_bp/execution/episode，group表按完整场景键统计。没有删除路径或缩减预算。

status.json的complete=true只表示scope=static_direct_benchmarks_only，不能当作整个多模型研究完成。记录28场景组、真实路径数、有序episode ID序列SHA、原路径SHA、所有父文件/当前源码SHA和输出SHA。开始重跑先标running；错误写failed且重新抛出。CSV流式写临时文件再replace，避免把全量日账本全部积存在内存。

## 共享金融执行

同一真实路径仍来自原read_paths。所有场景复用direct_dynamic.rollout，最后调用唯一原contract_ledger.ledger；传入ratio显式Python float，避免NumPy2 NEP50导致ratio×整数单位仍为float32。原premium、实际strike/unit调整、登记股息权益、除息应收、支付和期末期权结算/现货清算全部保留。

下一收盘不是简单把ratio向后平移：当日ratio×当日unit成为绝对股数pending；下日先成交该绝对股数，再按下日信息决策；最后一日取消pending并直接结算清算。记录当日raw_decision、实际executed_ratio、提交pending_absolute_shares、决策clip、单位界限fill clip和到期取消股数。

各定义的原始动作只依赖当日quote、当前/过去可得曲面与历史RV，按唯一当前报价键缓存，四个费用/执行场景共用同一动作，不重复曲面拟合或按成本调Delta。BS仍优先从当前真实option quote求IV，报价不可识别时用当日RV20；不再把上一条评价路径的IV带入当前报价。

path_pnl包含归一化终值、真实成本、包括期末平仓的换手、clip及回退决策数。group_metrics包含同场景路径数、均值、RMSE、标准差、分位、下尾5%均值、平均成本/换手、clip总数，与动态直接基准使用同一汇总函数。

## 首日stale和RV20因果修复

旧实现只遍历实际评价日期，因此首个评价日前已合格曲面及没有评价路径的中间交易日都可能被漏掉。现在按完整排序现货交易日遍历，从首评价日前最多5个交易日开始，到最后评价日结束；只接受status complete且model.diagnostics.valid的曲面。当前日无合格模型时使用过去最近≤5交易日状态，超出则RV/BS；不读取未来曲面。逐日记录来源日期、stale_days和回退类别。

原始ETF日线先sort_index，RV20与共同warm一致地使用 `log((S_t+当日现金股息)S_previous)`，20条历史总收益的年化平方均值开方。只用截至当前日的信息；连续收益率和融资仍为0。RV只是波动率输入，账本仍支付真实登记权益股息，两者没有互相替代。保留原预设最低fallback年化sigma=.01，未根据评价表现调整。

合格曲面的Black IV不可识别或Delta数值异常时沿用原RV/BS回退并记录；不改变冻结曲面参数。有限曲面/外推性质仍由原模块诊断决定，本次未宣称全域无套利。

## 测试证据

先新增4项回归并运行旧实现，全部失败：缺少新因果helper/延迟执行接口，以及旧run没有完整场景完成receipt。修复后4项全部通过，约1.363秒。

命令：`工作区/运行支持/环境/volatility/Scripts/python.exe -m unittest discover -s 工作区/当前研究/程序/波动率模型比较 -p test_direct_static.py -v`。

- 手算登记/除息/unit100→102：延迟执行第一日0.5对应50股，下日仍成交50股，绝不是51股；到期取消最后51股pending，终值wealth=3.27。
- 首评价日2020-02-01能读取此前2020-01-30合格曲面，stale=2个fixture交易日，而非直接降级RV。
- 现金分红恰好解释除息落差时，调整总收益RV不受除息机械跳跃污染；任意改动后日价格不改变当前日RV。
- 临时两真实路径完整端到端：28个场景组、56条pnl、168行日账本；每个场景组episode集合均为{1,9}，状态及输出SHA成功。

这些是共同财务和场景完备性验证，不是真实四折绩效或模型优劣结论。全量仍须由root协调并读取scope与status，不因文件存在就当研究完成。
