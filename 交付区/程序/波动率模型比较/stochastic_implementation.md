# Heston / SABR 随机波动率模块：接口与数值验证

新增 `stochastic_volatility.py`、`test_stochastic_volatility.py` 和本报告；没有修改既有曲面模块、行情、环境或交付区。没有运行真实行情训练或全量实验。

## 接口

所有定价均 r=q=0、spot=forward=1，t 单位为年，k=K/spot，call 返回 call/spot。数组可广播；t≤0 返回内在价值。

| 函数 | 作用及结果 |
|---|---|
| `price_heston(t,k,v,kappa,theta,volvol,rho)` | Heston 归一化 call；v 为当前方差，返回 numpy 数组/零维数组 |
| `price_sabr(t,k,alpha,nu,rho)` | beta=1 的 Hagan 近似 call；alpha 为当前波动率；拒绝 rho>0 |
| `price_model(kind,model,t,k,state)` | 上述模型统一分派，kind=`heston` 或 `sabr` |
| `fit_global(kind,t,k,prices,states,seed=20261112)` | 给定已因果计算的 states，五初值拟合共同 Q 参数，等权归一化价格 MSE，返回 JSON 字典 |
| `filter_state(kind,model,t,k,prices,previous=None)` | 仅使用调用方传入的当前截面，单维有界价格 MSE 拟合，返回 float；previous 仅作为额外比较候选 |
| `check_q_grid(kind,model,state,times=(.02,.1,.3,1.))` | 当前固定状态下的 call 价格上下界、执行价单调/凸性、日历单调有限网格诊断 |
| `estimate_p_transitions(kind,states,next_states,dt,spots=None,next_spots=None)` | 仅从显式训练转移估计 P 漂移；dt 可逐笔不同且以年为单位，返回 dict |
| `simulate_p(kind,model,p_parameters,spot0,state0,dt,shocks)` | 输入共同的独立标准正态 shock 数组 `[...,steps,2]`；返回包括初始点的 spot/state numpy 路径 |
| `q_derivatives(kind,model,t,k,state)` | S=1 下 price/delta/gamma/state_derivative/state_second/spot_state_cross/lmv |

Q 参数字典：Heston 为 kappa/theta/volvol/rho；SABR 为 nu/rho（beta 固定为 1）。fit_global 会附 kind、diagnostics，定价函数只读取对应数值参数。

P 漂移参数字典：Heston 为 state_kappa/state_theta/spot_drift；SABR 为 state_log_drift/spot_drift。经验扩散和相关估计仅保存在 diagnostics.state_volvol_empirical/rho_empirical。state_log_drift 是 logalpha 的年化漂移。模拟直接使用该漂移，避免经验扩散与冻结Q扩散不同产生漂移单位偏差；兼容旧caller的state_drift时，将其明确解释为冻结Q扩散下的alpha算术漂移，并减去半个Qnu平方。未提供现货转移时 spot_drift 默认 0。研究实际现货风险溢价时应传入训练现货转移，不应把默认 0 解释为实证估计。

## 定价、估计与模拟

Heston 使用稳定的 Little-Heston-Trap 特征函数，以 256 点 Gauss-Legendre 对 P1/P2 反演。每条报价积分上限为 `clip(40/sqrt(E[integrated variance]),200,2000)`。volvol<1e-5 时采用确定性积分方差的 BS 极限，不存在强行设 v=theta 的错误。没有通过裁剪 call 价格隐藏积分误差。文献：[Albrecher 等, The Little Heston Trap](https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/HestonTrap.pdf)。

SABR 使用 Hagan beta=1 的 lognormal IV 渐近式及 ATM 小 z 展开，再代入 BS；这是近似定价，不是精确 SABR 条件期望。nu=0 精确退化为常波动 BS。rho 限制在 (-1,0]，校准限制在 [-.95,0]，排除 beta=1 正相关导致的严格局部鞅问题。若渐近式产生非正波动率，直接报错；网格诊断将此情况记为 valid=False。文献：[Hagan 等, Managing Smile Risk](https://www.stat.ncu.edu.tw/teacher/wenteng/2010%20fall%20seminars/20100929%20%E6%AD%A3%E9%89%89/sabrAll.pdf)。

全局 Q 校准与当前状态拟合分开。Heston 全局边界：kappa [.05,15]、theta [.0025,1]、volvol [.02,2]、rho [-.95,.5]；SABR：nu [0,2]、rho [-.95,0]。固定 seed、五初值、L-BFGS-B；第一个初值为常用中值，其余按给定边界和种子生成。两模型价格目标相同，返回的 objective 是原始 MSE。状态滤波界：v [.0004,1]、alpha [.02,2]，无未来平滑。函数无法验证外部传入 states 的生成方式，训练截止日期和截面顺序由调用层负责。

Heston P 漂移用 `dv=kappaP(thetaP-v)dt+xi sqrt(v)dW` 的 Euler 加权最小二乘估计，权重 1/sqrt(v dt)；kappaP∈[1e-4,30]、thetaP∈[1e-6,4]。SABR 用 logalpha 转移估计比例漂移和残差方差；现货漂移从对数收益加半方差修正估计。传入转移必须属于合法训练期、确实相邻，不能跨越留出窗口拼接；至少三条转移。

**严格等价测度下扩散系数和 rho 在 P/Q 间必须共享。** simulate_p 已实际强制使用 Q model.volvol/nu 与 model.rho；P 输入只提供漂移，经验扩散/相关估计仅供诊断。即使调用方额外传入旧式 state_volvol/rho，也不会覆盖冻结的 Q 扩散。新增回归测试先重现旧行为产生不同路径，再验证修复后路径逐元素相同。

P 模拟：spot 使用对数 Euler；Heston 方差使用 full-truncation Euler（每步非负截断），SABR alpha 使用对数精确一步。第二因子 shock 为 `rho*z_spot+sqrt(1-rho²)*z_independent`；不在函数内调用随机生成器，因此所有竞争模型可输入完全相同的 shock 数组。离散误差须在真实实验中用时间步收敛检查，当前小测试不能替代该检查。

Q 导数通过中心差分计算 S=1 下未归一化价格 `S*price(t,K/S,state)`。输出 lmv 为 `delta+rho*xi*C_state`（Heston xi=volvol；SABR xi=nu），对应共同扩散下的瞬时局部最小方差投影，不依赖 P/Q 漂移。

## 数值验证证据（2026-11-12）

先创建测试并运行：实现文件缺失，6 项全部因明确的模块存在断言失败。实现后增加 Heston 全局拟合/Q 路径矩测试；另先重现非正 Hagan 区域使网格诊断异常退出的问题，再修复为诊断失败而非冒充成功。

最终执行：`工作区/运行支持/环境/market/python.exe -m unittest discover -s 工作区/当前研究/程序/波动率模型比较 -p test_stochastic_volatility.py -v`。

结果：**10 项通过（新增SABR log漂移单位回归），无警告**。测试覆盖：

- 两模型常波动极限、Heston 到期内在价值；
- Heston Gauss 定价与独立 scipy.integrate.quad 积分，在 T=.7、K/S=1.1、v=.04、kappa=1.5、theta=.05、xi=.4、rho=−.6 下对照到小数点后 7 位；
- 价格范围、执行价凸性、Delta 与独立现货价格差分相符；
- 仅传当前报价的因果状态拟合、未提供未来数据不会读取未来；
- SABR P 扩散残差估计、显式共用 shock 的路径可重复性、非法 dt 和 SABR 正 rho 拒绝；
- 两模型 Q 参数合成样本校准；无效 Hagan 区域返回 valid=False；
- 20,000 条 Heston Q 条件路径、64 步、T=.25：mean spot=1.0011702（理论 1）、mean variance=.0395976（理论 .04）；ATM MC call=.0393923、CF call=.0390812，误差 .0003111，落在测试的 Monte Carlo/离散容差内。

补充 2 期限、10 报价的合成共同价格目标校准：Heston MSE=2.2152e-18、约 .968 秒；SABR MSE=0、约 .013 秒。两者有限网格均 valid。这只是定价和接口验证，不是完成真实行情训练，也不支持实证表现结论。

## 局限

网格 k∈[.5,1.5]，期限包含输入期限和 .02/.1/.3/1 年以及零，固定状态取训练状态最小/中位/最大分位点；容差 2e-6。只宣称 finite_grid_only，不将有限检查当作全域证明。Heston 极短期、极端执行价、边界参数仍可能需要更高积分精度；SABR 大 nu、长到期和极端执行价可能出现渐近式静态套利。当前全局优化为局部多初值，没有全局最优证明，也不保证参数在稀疏报价下可识别。

没有真实行情清洗、利率/股息校正、在线日期索引校验、P 训练截止截断、真实模型训练或任何交付区晋级动作。调用层必须保存冻结的参数、训练边界、合法转移账本及数值诊断，诊断失败时停止下游结论。


## Heston 短期低方差真实反例修复

2026-11-12 真实校准候选 kappa=.459383058744558、theta=.13825085092212847、xi=.5325879972389376、rho=−.045780834961336114，在重建最小训练ATM代理方差 v=.002877310695567458、T=1/365、K=.5..1.5 时，原256点有限Gauss产生约−.01760的虚假call下界违背，属于数值积分失效，不能用来淘汰理论Heston模型。新增回归先确实失败后修复。

定价现在检测高振荡 upper*|logK|>60，以及非有限/价格界限异常。相应报价转为Lewis阻尼表达式，使用scipy QUADPACK带cos/sin权重的无限Fourier尾积分；ATM采用自适应无限积分。没有裁剪call价格。pricing_diagnostics(reset=False)返回quotes、fallback_quotes、fallback_fraction及fallback_max_error_estimate；fit_global带quadrature快照。计数是进程累计值，调用方可在任务前reset以获得单任务统计。

同一真实参数在期限[1/365,.01,.03,.1,.3,1]及161执行价网格现valid：call_lower约−1.19e-12、strike_monotonic约−1.84e-11、strike_convex约−1.94e-10，已消除原伪套利。该真实失败版本应归档为旧数值实现产物并重新拟合，不沿用原valid=False结论。回退显著增加深尾网格/极短期成本；真实训练仍需保留回退率。


## 2026-11-12 全局Q五候选选择器修复

旧fit_global先按optimizer success/价格loss只选一个候选，再检查所选模型的Q网格。SABR Hagan近似可能出现最低loss成功候选在长到期grid失败，而其他较高loss成功候选通过全部grid；旧流程因此错误拒绝整个模型。

现保持五个初值、随机数、L-BFGS-B、bounds、objective、全部定价/quadrature/状态/P漂移/模拟函数不变，只在全部优化完成后逐候选检查原Qgrid。state取原样本最小/中位/最大分位（相同值去重），期限和执行价格点与原定义完全相同。qualified要求优化成功、有限objective和所有固定state Qgrid通过；只在全部qualified中选最低原价格MSE。没有qualified时返回最低失败MSE候选用于诊断，valid保持false。

每项attempt保存参数、objective、success、optimizer status/迭代/消息、完整grid及grid_valid/qualified。顶层追加qualified_candidates、0-based selected_start、grid_state_quantiles和selection_rule。quadrature仍是原进程累计计数，仅因所有候选都做网格验证而记录更多真实定价调用，没有改积分算法或阈值。

新增3个mock optimizer但真实金融价格/Qgrid回归：高nu=2/rho=−.95在短期能拟合而1年非正Hagan的候选，不得遮住nu=.3/rho=−.4且三个state分位均合法的候选；全部grid失败即使optimizer success也必须invalid；低loss optimizer失败不能胜过成功合法候选。先跑旧实现出现1项错误选择失败和2项缺新诊断字段，再修后14项全通过，约7.229秒。

对root的OUT/attempts/pre-global-selector/stochastic_volatility.py snapshot核验：选择器之前源码prefix逐字相同，minimize调用AST完全相同，除fit_global外其余14函数AST全部相同。root可据此独立证明旧valid最低loss候选仍属于新qualified最优集合，迁移已完成校准；本次没有自行修改任何模型产物、重跑真实校准或更改旧验收结论。

仅修改stochastic_volatility.py、对应测试及本报告，修复后不再扩充功能，等待scope review。
