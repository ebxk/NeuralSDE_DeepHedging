# 三层静态价格族比较报告

仅新增surface_report.py、test_surface_report.py和本报告；未改动任何模型kernel、策略、基底或训练生产器，没有运行真实全量汇总或重新拟合。

## 接口与产物

`build_report()`，或`volatility/Scripts/python.exe surface_report.py`，默认四折、10种静态价格族：SVI/SSVI/eSSVI native，Heston/SABR，original_grid、nsde_factor、svi_factor、ssvi_factor、essvi_factor。

输出 `OUT/evaluation/`：

- surface_daily.csv：每fold/date/view/method/domain的聚合SSE、n、valid/algorithm_valid、失败/缺失/blocked标签、来源路径和SHA，**不含逐条原始报价**。
- surface_comparison.csv：按训练/验证/历史评价、full/heldout及domain汇总的有效日期/失败/覆盖率、price和IV SSE/n/RMSE、all-attempt及共同日期/共同IV位置的matched结果。
- quote_errors.csv.gz：本地逐报价重建误差，含code、ttm/k、observed/predicted normalized call和IV、低Vega/不可识别/共同qualified标记，不可上传Git。
- surface_status.json：完整source覆盖状态、合法blocked的missingmethods、source_errors、所有父文件/输出SHA、范围/容量/外推说明。只有source覆盖完整（允许严格可核验blocked）才complete=true；缺源、错view、不可重建状态都false。

**Git只允许surface_comparison.csv聚合表。** daily和quote errors保留本地颗粒，不能把全部日期/逐quote文件推到Git。

## 冻结参数重建，而非引用泄漏的汇总数

native读取各日full或heldout.json的对应model，并核对identity.day/kind/view；直接price_surface重建当日每条q价格。绝不把full.json的hidden RMSE称作留出误差，也不把其visible=all统计与hidden再相加重复计数。

Heston/SABR从当前fold global.json、states.csv分别读取state/full和masked_state/heldout，经冻结Q定价重建；原fit_rmse/held_rmse只属于已有诊断，实际报告误差和计数均从同一q及自身重建价格计算，不用另一套计数口径。

factor的daily.csv提供有效标记，daily.jsonl提供保存的xi或original grid。basis/center冻结，curve使用scale=np.ones(2)；original以保存grid作center、zero basis查询。nsde_factor读取新clean original encoder，其他factor读取各自linear basis。CSV/保存状态有效标志不一致或重复view/date会拒绝。

full评分使用该日全部共同q，是描述性重构误差；heldout仅评分q.holdout=True，并必须用对应独立masked状态。训练/验证/历史分割严格沿用fold截止年，不据结果重选日期。

报告过程没有任何minimize、fit、repair、basis训练或市场再拟合。NN drift不改变静态price family；两因子压缩与native多期限参数容量不同，不能把静态误差差异归因于动态NN。

## 价格、独立IV和期限/翼部

价格单位始终call/spot，所有汇总用 `sqrt(sum(SSE)sum(n))`，不用日RMSE等权平均。独立IV是本模块自己的48步向量化BS二分，r=q=0、年化sigma∈[0,10]，不修改任何已有定价kernel。

市场和模型各自反解IV、计算归一化Vega；任一价格不在可识别区间、超反解上界，标unidentifiable。任一Vega<1e-4，单列low_vega，不计IV误差。价格误差仍保留，不让低Vega筛选删除price样本。所有IV统计都给IVSSE/n与排除计数，不编造缺失IV值。

五个domain：all；in_domain=T≤90日且k∈[.9,1.1]；short=T≤14日；wings=k<.9或k>1.1；out_of_tenor=T>90日。它们**重叠且不能相加**，用于明确原网格14/30/60/90天及legacy外推与native期限切片不同带来的能力差异，不隐藏域外劣势。

## 公同qualified样本，防剔失败赢家偏差

每个fold/date/view同时处理所有可用方法。除严格合法blocked的整方法外，任一方法该日不合格或缺源，该日不能进入任何方法的matched价格比较。即使某方法所有日期失败，也不把它从共同交集中删除来制造赢家；共同日期可为0，RMSE为未定义空值，绝不以0误差冒充。

matched IV在相同共同日期之上，再取所有方法**共同可识别且高Vega的逐quote交集**，保证IV比较也同分母。blocked方法本身matched_n=0、RMSE空值；不会借别的方法有效性得到虚假的matched数值。单方法qualified、all-attempt和共同集合都分别保存，valid/失败日与coverage明确。

合法blocked沿用acceptance_manifest的当前generation/source生产器证据规则。Heston/SABR数值生产器阻断可解释missing方法；压缩factor只有确实缺basis且该新SVI支路合法blocked才可排除。已有basis但未执行factor audit不能因动态生成器失败就自动视为科学完成。

## 来源和完成状态

所有读取文件记SHA，既有parents/sources/output_sha声明经共同Auditor核验；native还核quotes/runner/source及view。报告末再次重哈希全部父文件，防止长时间重定价中混入不同代产物。只写本任务新报告，不覆盖任何校准回执。

source缺失仍写surface_status.complete=false及错误原因，不把未执行当failed fit。完整报告可以含已量化numerical_rejected/fit_failed行，但其valid=false，不能悄悄从原始all-attempt表消失。

## 测试证据

先写4项测试，模块缺失时全部失败。最终6项小fixture测试通过（约1.33秒）：

1. native full文件的hidden RMSE故意写0、heldout参数sigma=.3；报告仍从heldout模型重建隐藏报价，得到真实非零SSE，full泄漏汇总不起作用。
2. 两方法不同有效日期时，共同qualified日期和quote数完全一致；SSE/n严格加权（1与9条报价的SSE分别1/81，RMSE=sqrt(82/10)）。
3. 独立IV恢复BS .2、非法价格不可识别；short/wings/long domain判定。
4. 空目录缺来源明确failed并写状态，不假报complete。
5. 高可识别但低Vega报价不计IV误差；合法blocked方法没有虚假matched数值。
6. 实际SABR定价的full .2/masked .3保存状态选择，以及factor full/masked保存xi分开读取，均真实重建不同条件价格。

命令：`volatility/Scripts/python.exe -m unittest discover -s 工作区/当前研究/程序/波动率模型比较 -p test_surface_report.py -v`。

未运行实际四折全量重定价，因此本任务没有报告真实方法优劣、共同日期数或新的生产complete凭据。全量可能受Heston自适应积分与大量日/quote计算影响耗时；程序采用逐日/流式gzip和小聚合状态，避免保存全体raw quote误差于内存。所有必需IV、期限、翼部计算已实现，未用“现有汇总恢复不了”替代必交付分析。
