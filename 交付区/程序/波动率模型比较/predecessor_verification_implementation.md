# 冻结前序缓存全预算等价验证

仅新增verify_predecessor.py、test_verify_predecessor.py和本报告。验证器不拟合/保存模型权重，不修改旧source、market.done.json或任何旧输出。只写新研究 `OUT/predecessor/end.json` 证明/失败凭据。

## 接口和完成条件

`工作区/运行支持/环境/market/python.exe verify_predecessor.py --end 2025`，或`run(end)`。

旧marker记录的market.py SHA与当前版本不一致，不能凭文件存在或小样本相近就复用。验证依次要求：

1. 旧marker.outputs的所有7个文件逐一SHA一致，包含common/single/pool、market.json和三个weights。
2. 旧marker里的C1.npz、episode_days.csv、集中ETF原始CSV输入SHA一致，配置字典完全一致。
3. 原conditional_market.load_frozen冻结加载器仅将hardcoded外部ETF路径的一行替换为PROJECT集中副本，且该副本SHA必须等于旧marker。记录原loader文件/函数SHA及适配函数SHA。其余作用域和冻结加载流程完全复用，未改旧文件，也不再依赖外部quant数据库路径。
4. 冻结加载器内部原有real_xi与real_cov严格核验必须通过，读取的market模块必须是当前工作区指定源文件；实际2026权重SHA再次与marker核对。
5. 原market.prepare_origins重建全部40960个起点，按原种子104、原episode分组顺序；time/warm/strike/premium按照原存储dtype cast后逐元素exact。模拟保留原float64时钟，不用共同文件四舍五入后的float32时钟。
6. shock_rng105一次生成完整40960×(H−1)×4×3正态数，按原四→二子步聚合；原market.simulate、subdivisions=2、boundary=reflect，全路径重放，不抽样。
7. spot=exp(logspot).astype(float32)、xi=states[...,1:].astype(float32)，与旧single.npz所有元素逐一exact。任何一元素不同均failed，**不使用allclose宽容度，不以金融界限/策略表现例外放宽**。
8. 模拟后再次验证所有旧输出及marker SHA未变，源文件SHA未变，才complete=true。

最终证明准确包含供common_inputs消费的complete、current_source_sha256、marker_sha256；另有marker_expected_source_sha256、cached_all_output_hashes、loaded_weight_sha256、copied_spot_sha256、loader身份、全部common/模拟数组维度、元素数、最大绝对/相对误差和内容SHA、随机数数组SHA、耗时及原模拟金融诊断。status只有全exact才为exact_full_budget_equivalence。

数组比较SHA包含dtype/shape和全部字节；相对误差只作描述，分母floor=1e-30，不参与接受规则。非有限值或形状改变直接拒绝。实际模拟没有另存一份“新single”来替代旧结果，失败时只写新failure证明，不覆盖旧缓存。

## 测试证据

先创建3项测试，在模块不存在时全部明确失败。实现后market环境3项通过，约.032秒：

- 40960路径fixture中只改最后第40959条路径的末尾一个因子元素，完整比较明确拒绝，different_elements=1、first_difference_index=[40959,2,1]；该路径不可能被前16条小样本检查覆盖。
- 原float64数值cast回原float32后完全相等时通过，SHA一致；省略后半路径或改变shape拒绝。
- 全marker outputs SHA校验，任一文件改字节就拒绝；loader只允许精确一条spot路径替换，源码结构变动时guard失败。

命令：`工作区/运行支持/环境/market/python.exe -m unittest discover -s 工作区/当前研究/程序/波动率模型比较 -p test_verify_predecessor.py -v`。

## 真实2025验证

已按本任务授权启动全预算验证；最终实跑结果与耗时见本报告后续记录及 `OUT/predecessor/2025.json`。该步骤是冻结权重重放验证，不是重新训练，也不是抽样模拟晋级。


### 真实实跑最终结果：严格拒绝复用

2025折全预算实跑已完成，原simulate耗时20.9703941秒，总验证30.2640421秒。结果为complete=false、status=failed；新证明保存在OUT/predecessor/2025.json，没有用容差把它改成成功。

| 数组 | 完整形状 | 比较元素数 | 不同元素数 | 最大绝对误差 | 最大相对误差 | 首差索引 |
|---|---|---:|---:|---:|---:|---|
| spot | 40960×65 | 2662400 | 53 | 2.384185791e-7 | 1.050919134e-7 | [40714,57] |
| xi | 40960×65×2 | 5324800 | 991 | 9.536743164e-7 | 5.043264895e-5 | [40626,60,0] |

差异集中在末尾少量路径，小样本前16条检查确会漏掉。旧source SHA=d3698adfe717fac5d6908df2f5621470bc2cd8c37ac4ffdfce23f3067d867853；当前source SHA=870cd7cda5cb13f94e87eee24467841d67473ef4f9c9b3f351982318be604afc。旧marker SHA=929c30731b992d26b380b07a81e53d0f2593e67015271b62dd4c0cc8527f2477。

运行前7个旧output SHA、原输入及集中spot SHA、冻结real_xi/real_cov核验、全部common time/warm/strike/premium均通过。模拟失败后另行只读复核7个旧output及marker SHA仍完全不变，确认未修改旧缓存。

当前严格拒绝不能被common_inputs当作source等价授权；如需采用实际政策财务误差判据，必须另行明确设计和验证，而不是把本次非exact结果重标为成功。当前程序没有新增金融容差例外或实际policy接受分支，不能以“误差很小”冒充已证明。

运行环境出现既有TensorFlow缺GPU DLL告警后使用CPU，以及既有pkg_resources弃用告警；没有安装GPU、修改环境或屏蔽数值失败。最后fixture fresh复跑3项全通过(.015秒)。
