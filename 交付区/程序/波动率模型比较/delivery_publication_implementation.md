# 交付与公开发布实现说明

本次只扩充 finalize_delivery.py、新增 publish_delivery.py 与对应测试，未改金融内核、冻结参数或实验结果。finalize 仍先调用严格 acceptance manifest 验收，再检查训练权重/父代哈希、实际重载评分和账本验收。当前验收覆盖由 artifact_guard 的 expected stage keys 决定，不在交付脚本中放宽。

本地交付复制策略/市场网络权重、因子 geometry、Q/P 参数、状态和身份回执。inputs、模拟 simulation/reference_common、预测逐起点数据、因子逐日及逐报价诊断留在工作区原位，以 local_artifact_references.json 记录绝对位置、字节数和 SHA256，避免重复复制数 GB 路径。清洗公共曲面与冻结配置、依赖保留本地。直接对冲、预测、数值诊断新增聚合表；NSDE 直接对冲包含标准差、CVaR95 损失、平均换手。

公开发布使用从远端 main 创建的独立 checkout，只暂存本轮程序、冻结配置/登记表、可安装依赖版本、明确聚合表白名单和已有 tradeoffs 图。surface_comparison.csv、surface_status.json、numerical_diagnostics.csv、compute_costs.csv、numerical_status.json 均已纳入。拒绝 raw、逐路径、逐日、权重和 npz；JSON 的路径键、路径值和嵌入说明脱敏，中文相对路径保持不变。每个公开文件与本地特殊变体均记录哈希。

旧 verify_predecessor.py 的公开版本仅保留 reference_market 所需的数组等价比较 helper，AST 核对保证 helper 不变；不发布含旧机器路径的过时验证 CLI 测试。工作区原文件不改。依赖锁中的 @ file 构建路径仅从对应既有环境实际安装元数据转换为 package==version，无法确定版本会拒绝，不猜版本。

publish_delivery.py 默认准备独立提交，只有 --push 才作非 force 推送；所有验收及文件脱敏预检在 clone/commit/push 前完成。本次未执行发布函数，未 clone、commit、push，也未启动全量实验。

验证：先将“大模拟样本不得复制”的回归测试运行至失败，再实现按参数类型复制和原位清单。8 项 unittest 全部通过，覆盖验收 gate 优先、局部参数复制、尾部/换手手算、公开白名单、逐路径拒绝、JSON 脱敏、中文相对路径保留、原 helper AST 与依赖固定。另对当前全部可公开程序执行纯文本路径预检，无失败；没有读取或改写科学结果。
