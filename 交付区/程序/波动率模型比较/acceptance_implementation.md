# 固定完整研究范围的验收manifest（I5）

仅新增acceptance_manifest.py、test_acceptance_manifest.py和本报告；未调用任何生产阶段、训练、评价或全量预测。

`expected_stage_keys()` 返回固定批准范围的998个唯一有序字符串键。包括四折、9生成器、432策略及逐策略四费用/执行评价；共同清洗准备和新clean-base、四折原编码/新NSDE参考、15876曲面记录、773路径财务验收、静态/NSDE/动态直接基准、20个因子审计、36个预测分支以及完整统计/区间/报告。不会由现存文件反推出一个缩水的“已完成范围”。

`build_manifest()` 只验收已保存产物，原子写 `OUT/acceptance_manifest.json`，字段complete/pending/stage_errors/declared_failures/coverage/artifacts。缺文件逐项pending，不会因缺表抛出后不写manifest。CLI未完整时exit1。coverage每项记录status、来源绝对路径及SHA；global artifacts覆盖所有引用证据和当前验收器源码，供artifact_guard完整键集比较及重哈希。

## 关键验收

- 数据：2646日期、共同报价有效K去重、原数据SHA、每折episode源SHA；registry严格432唯一完整坐标与当前CONFIG。
- 新clean-base：data/clean_base/status.json、共同shared_surfaces.npz的2646×20、金融域；每折representations/end/original.json及original.npz的来源/训练截止/二维20网格/金融可行性。新reference.factor和所有成功ready父基底必须指向此新original编码，旧pilot原C1不能冒充共同质量规则代。
- 曲面：stage_complete、结果数15876、2646×3模型×full/heldout的所有身份和实际记录存在；source/runner/quotes SHA；仅量化numerical_rejected可声明科学失败，脚本failed不是科学结果。
- 生成器：每折新Dense64/Dense64/Dense7、当前source/parents/CONFIG、40960预算、7槽f、latent sidecar及所有既有SHA核验。reference必须是fresh fit，不能旧weight复用。
- 策略：432个任务全部为可验证weight/initSHA的trained receipt，或具有当前ready/trainer/policy精确context和numeric category的实际训练失败。缺训练或旧generation失败不能终结。
- 财务：773路径、独立账本和旧权重重载误差均有限且≤1e-8，含股息/调整/nextclose。
- 直接：每折static 28场景、NSDE 8场景、Heston/SABR各8和LV4场景，逐组同实际path IDs；日账本和group等实际文件存在。dynamic benchmark_missing只可沿当前已核验generator block链声明失败。
- 因子：五表征、每日期两视图、完整daily/summary/identity、按split/view报价计数、初值SHA、失败fit数量证据。缺新basis只可对应已明确阻断的SVI支路。
- 预测：每分支实际结果和1/5/20期限计算/排除账本；自身模型价格增量、有限CRPS、样本数。当前blocked生成器不可沿用旧complete预测。即使所有分支均声明blocked，也至少要有一项真实完成预测，否则整个预测阶段不能凭空完成。
- 评价：每个trained job四场景、相同path IDs/n、实际NPZ/path/day账本和1e-8误差；冻结selection精确匹配已训练/明确失败分区。overall/seed summary、5个分组维度分割、配对1/3/6月block且2000次、四折24个原C0/C1历史seed引用及其sourceSHA、报告行数/图和每trained job nextclose都齐全。

## 科学失败与基础设施错误

generator必须有该模型12个registered job的category=numerical_rejection blocked记录及当前source/parents可核验的真实生产器数值失败证据；仅traceback、旧pilot、缺字段、脚本异常/源损坏不能计作declared_failure。当前ready已重建时旧blocked不能压过新generation。

训练失败必须status=training_failed_current_generation、任务key一致，context精确等于当前ready/trainer/policy SHA；只有numeric category可声明失败。blocked依赖必须存在，禁止自循环/循环。所有stage_errors均令complete=false，与允许报告的量化科学失败分开。

若现有producer失败receipt没有source/source_sha256，协调器需提供带当前source的producer_evidence，或生产器补完整证据；本验收器不会凭单个旧错误字符串替它证明本轮已执行。

## 测试

先写4测试，在模块不存在时全失败。随后发现并先重现两项实现边界：stage wrapper的默认status覆盖科学失败、含lambda0.1文件名被with_suffix截断；修复后加入currentgeneration/infra区分回归。

最终7项测试通过，约.083秒：固定范围及空coverage拒绝；direct/predict/factor任一缺失阻交付；合法blocked保留失败标签且依赖缺失拒绝；空OUT仍写incomplete；哈希改变拒绝；真实量化失败能穿过stage wrapper；lambda0.1完整评价文件名；infra/stale训练失败拒绝而匹配numeric context可声明。

测试仅临时小fixture，未把fixture覆盖当作真实998项完成。真实当前研究尚未运行本脚本全量验收，未写任何complete=true生产manifest。后续新clean-base生产器须遵守此报告列明metadata字段；缺阶段仍会明确pending/error，不能生成“已完成研究”。
