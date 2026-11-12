# 同一质量规则下的原曲面公共基线

新增prepare_clean_base.py、test_prepare_clean_base.py及本报告。仅额外适配reference_market、generator_export、factor_quote_audit及对应测试的原基底/公共面路径；未修改root负责的common_inputs、linear_market、build_representation、direct_nsde、prediction_audit等脚本。

## 公平性口径

原始旧曲面曾在正价/量/持仓筛选后直接修复，而原生SVI使用更严格共同价格界限清洗，不能把两者的清洗差异当成SVI方法收益。新基线只从当前 `OUT/data/quotes.csv` 的同一批已清洗报价重建，SVI与原方法面对完全相同的质量规则。

旧surfaces.npz仅选择读取A、b、grid_t、grid_k；**不读取旧calls/raw/dates/spot**。旧每折C1.npz仅选择读取weights覆盖权重；**不借旧basis/xi或NN市场权重/模拟**。这些文件作为固定金融约束/覆盖权重的父SHA记录，不代表复用旧市场结果。

## 曲面阶段

命令：`volatility/Scripts/python.exe prepare_clean_base.py --stage surfaces`。

逐日full view复用factor_quote_audit.fit_original_view：可见报价原插值、0期限payoff/零执行价锚、原固定−.5受限右翼、固定A,b的scipy L1修复。不重新筛选一个不同报价集合；已有共同价格界限不合格输入会明确拒绝。

为保存原始重建值，fit_original_view仅增加raw_grid返回字段，其价格/修复/优化语义未改，factor扁平CSV排除此数组字段。产出 `OUT/data/shared_surfaces.npz`：2646日期/spot、raw与calls `(2646,20)`、repair_mean_abs/repair_l1/repaired、quote_outside_grid_fraction、strike_extrapolated_fraction、tenor_extrapolated_fraction，以及固定A,b与20网格。

`OUT/data/clean_base/status.json`保存complete/status、source/parents、quote_sha256、shared_surfaces_sha256、output_sha256、修复/外推和金融诊断。只有全2646天及金融约束通过才complete，临时NPZ原子replace，元数据最后写。

## 编码阶段与进程隔离

命令：`volatility/Scripts/python.exe prepare_clean_base.py --stage encoder --end 2025`。

入口通过subprocess启动market Python独立worker，避免PyTorch/TensorFlow/旧cvxopt运行时混用。子进程移除继承的PYTHONHOME/PYTHONPATH/VIRTUAL_ENV，强制使用market自身标准库；worker拒绝已加载torch的环境。

旧representation.decode保持原方法，输入是新shared_surfaces、训练日期mask(date<训练年末+1)及仅从旧C1读取的冻结weights。只允许一次 `w -> .5+.5w` shrink重试，不能进一步调参或放宽1e-7金融容差。输出前验证因子W/bounds与重建价格A,b均可行。

产出 `OUT/representations/end/original.npz` 和original.json；包含新的dates/spot/basis/xi/center/W/bounds及原decode诊断、train_end、shared_surfaces_sha256、parents/source/output_sha256。`original_weights_sha256`指旧C1文件的只读权重来源SHA，其basis和xi没有参与新拟合。

## 三组件适配

- reference_market读取`study_core.base_factors(end)`作原联合NSDE基底，identity.factor也指新文件；仍只从旧common读取共同合约条件，不加载旧NN。
- generator_export所有生成器的共同grid/日期基底改为base_factors(end)，新NSDE仍从本轮新reference simulation取xi，sidecar不进策略f。
- factor_quote_audit的nsde改用base_factors(end)，original未压缩参照固定金融数据改读SHARED_SURFACES。

原2025 prototype reference因基底改变已经不是当前reference，不修改它或早先attempt源码；必须由root后续归档并重fit。其他三折编码、四折NN、策略与报告尚不由本任务执行。

## 测试与实际结果

先写3项clean-base测试，模块缺失时全部失败；实现后通过。实际第一次编码启动遇到父Python3.13的PYTHONHOME污染market3.9标准库，明确失败，没有伪造编码结果。新增子进程环境回归先因缺env失败，修复后4项clean测试通过；为保持当前source身份，再完整重建一次约9秒曲面，内容SHA不变。

测试还覆盖：旧NPZ中calls/xi/basis为不可读取对象仍只加载允许字段；同一BS报价与价格界限拒绝、金融修复；mock decode明确只用训练mask，任意改未来calls不改冻结center；恰好一次shrink。generator/factor fixtures将旧basis/xi设为毒性数据而新original正常，实际导出/审计仍通过；reference路径测试在任何fit前捕获实际np.load，确认为新original.npz。

按授权已实际完成两步：

| 项目 | 结果 |
|---|---:|
| 曲面日期 / 报价 | 2646 / 156512 |
| 20-grid重建耗时（最终） | 9.5570535秒 |
| 修复天数 | 805 |
| mean normalized repair MAE | 3.218459526e-5 |
| 最小价格金融slack | −1.904467440e-15 |
| 平均strike外推比 / tenor外推比 | .1825207861 / 0 |
| 2025 encoder worker耗时 | 13.4567403秒 |
| basis / xi形状 | 2×20 / 2646×2 |
| 训练截止 / shrink | 2022年末 / false |
| 最小因子金融slack | 9.439174371e-10 |
| worker torch_loaded | false |

shared_surfaces SHA=`025f0e9893427b908264a107568e22e46969389816f4fea0960d61265426b6b4`；2025 original SHA=`d270f087fd57abbbdf5b021660843dbe3964794bafaeb07c62ec0cc1007e1052`。两步当前producer source均为`d4a4c6ee175549ef243d338d376d2db70e87346df3ab3eb41b74301f92cd41da`。

这是共同质量基线的重建与一个编码器验证，不是新SVI方法收益、不是新NN或策略绩效。金融修复和20-grid外推仍属有限域方法，未宣称连续全域保证。实际编码仅2025一折，其余折和所有NN/策略需后续执行。
