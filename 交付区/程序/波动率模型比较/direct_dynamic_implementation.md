# 动态直接对冲基准实施

新增 direct_dynamic.py、test_direct_dynamic.py 和本报告；未运行真实四折评价或训练。此前按明确追加要求扩展 generator_export 的诊断sidecar，未把latent/xi加入策略f或改变预算。

## 接口和执行

`run_model(end,kind)` 评价一折一个模型；`run(end=None,models=('heston','sabr','localvol'))` 在 end 为空时遍历 CONFIG 四折。CLI：`工作区/运行支持/环境/volatility/Scripts/python.exe direct_dynamic.py --end 2025 --model heston`。省略参数会实际运行全部四折，当前未执行。

必须使用含 torch 的 volatility 环境，因为原 hedge_experiment.read_paths 所在模块导入 torch。market 环境仅能执行不涉及 read_paths 的轻量测试；本次没有通过复制路径读取器规避共同路径定义。

真实路径严格由既有 `read_paths(原episode_days.csv,[end])[0]` 取得，既不删路径也不改ID。每模型记录原路径文件SHA与完整有序 episode ID 序列SHA，便于核验全部 benchmark 同一分母。

每模型输出 `OUT/direct_dynamic/end/model/daily_ledger.csv.gz`、`pnl.csv`、`groups.csv`、`status.json`。CSV/GZIP以临时文件写入后replace；status完成凭据最后写入，带父文件、实现源文件、输出SHA和数值/回退诊断。开始重跑先标running，避免旧complete凭据覆盖新的未完成过程。global Q产物缺失/全局数值诊断不合格则status=`benchmark_missing`、complete=false，绝不改成BS来冒充该benchmark。合格状态或LV曲面真正定价失败记numerical_failed并重新抛错。

## 冻结Q金融公式

Heston/SABR 直接调用 stochastic_volatility.q_derivatives：Delta为完整冻结Q参数下的现货偏导，LMV为 `Delta+rho*xi*C_state`。Heston state是方差v、xi=volvol；SABR state是波动率alpha、xi=nu，保持源模块单位。共同扩散在P/Q间一致，瞬时LMV不依赖P漂移。每个模型比较Delta和LMV，未根据评价收益重新校准参数。

日状态只用当天合格state；缺失时采用过去最近不超过5个真实交易日的正有限state；超过5日改为该日历史20个总收益计算的RV/BS Delta，LMV同BS Delta，并明确记录fallback_source/source_date/stale_days。此回退只针对日状态不可用，不针对全局模型无效，也不掩盖合格state的数值错误。原始ETF日线先sort_index，历史收益按实际除息额修正，未读取未来收益。

LV每天以当日eSSVI为新起点建120天表，time=0、spot_ratio=1、tenor=该日真实剩余期限、strike_ratio=该日实际K/S，使用真实条件PDE的Delta。每同日同到期的多执行价共用一次多右端求解。局部波动是一因子模型，其瞬时LMV等于Delta，因此只列Delta一条而不复制同一基准。

LV沿用已登记可用性回退：当日合格full curve→过去最近≤5交易日合格curve→该日历史RV20严格flat curve。使用旧curve只作为过去已知状态，没有读取未来curve。选中合格curve后Dupire或PDE失败不再回退，明确拒绝。每天处理完释放PDE缓存。

## 共同财务与延迟执行

所有路径、费用情景最终都调用原 contract_ledger.ledger，登记权益、除息应收、派息到账、单位与执行价调整、期末支付期权负债、清算现货完全由同一账本处理。CONFIG融资利率和连续收益率当前为0，没有私自新增利息假设。

成本情景为0/5/10bp同收盘，以及5bp下一收盘。`rollout(rows,decision,cost_rate,delayed=False,events=None)` 提供因果回调 decision(row,index,held_ratio)。延迟情景：当日提交 `clipped_ratio*当日合约单位` 的绝对股数，下个收盘先按绝对股数成交，再据当前日信息决策；合约单位变动不会把旧pending错误地重新乘新单位。到期先取消pending，直接结算并清算，不做“成交后立即反向清算”的虚假双交易。

Delta/LMV统一截至[0,1]，记录decision_clip_count。若执行时单位界限变化使绝对pending超过可持有单位，记录pending_fill_clip_count；真实单一90天合约常见单位调整为10000上调，手算测试确认没有把50股错误调整成51股。回调只收到当前row与实际持仓比率，没有未来row接口。

日账本附raw_decision、executed_ratio、pending_absolute_shares、回退来源；逐路径pnl包含归一化终值、成本、双边含期末清算换手、clip次数、取消的pending股数、term/moneyness。groups给路径数、均值、RMSE、标准差、5/95分位、底部5%均值、平均成本/换手和clip总数。收益全除以实际起点spot×起点单位，保持共同财务口径。

## 测试证据

先写4项测试，在实现文件缺失时全部因明确断言失败。实现后加入临时两路径端到端评价。最终6项测试使用volatility环境通过（含NumPy2财务精度回归）：

- 正常数BS波动率的next-close手算：首次不持仓，第2收盘才成交第1日Delta，期末取消最后pending；终值逐项等于初始premium+实际持仓价格差−期权支付−两笔成本。
- 登记/除息/支付/合约单位100→102的手算：同收盘0.5比例终值wealth=3.421且股息到账2.55；next-close执行旧绝对50股、登记日未持有不获股息、终值3.27，期末取消51股pending。
- 真实Heston/SABR非零volvol参数的Delta和LMV逐项匹配源Q导数；越界hedge动作统一clip并计数。
- 过去state最多5交易日、未来已有state不可读取。
- 临时两个真实路径ID[11,23]，SABR两hedge×四费用/执行共16条pnl及48条账本；真实LV PDE四情景同路径ID且Delta接近平坦BS极限；global改为invalid立即benchmark_missing，未发布替代BS模型结果。

命令：`工作区/运行支持/环境/volatility/Scripts/python.exe -m unittest discover -s 工作区/当前研究/程序/波动率模型比较 -p test_direct_dynamic.py -v`。

这只是小样本金融/接口验证，不是四折真实表现，也没有输出任何实盘优劣结论。极短期限LV Delta仍受有限网格精度影响；Heston短期深尾可能触发慢速自适应积分。全量评价应读取status和诊断，不能因文件存在就当已验收。

## 追加诊断sidecar

generator_export现额外atomic写 `sim_model.state_sidecar.npz`，ready里的state_sidecar包含file/sha256/complete。每模型包含origin_dates、episode_id、selected_origin_index；随机波动模型另含latent `(N,H)`，线性模型含原simulation里的xi，LV含逐路径origin_surface_hash。策略训练主npz、f和预算均不变。

旧ready若没有state_sidecar字段或sidecar SHA不匹配，**不能据此宣称完成1/5/20条件诊断**，须重新导出真实sidecar。新增回归先因缺少sidecar失败，再验证latent起点、episode ID、文件SHA和atomic无tmp通过；原5项export测试仍全部通过。


### NumPy 2 / NEP50 共同财务精度

rollout在把ratio送入原ledger前明确执行list(map(float,ratios))，费用也转为Python float，保持既有Python双精度现金口径。即使回调返回np.float32，不能让ratio*整数单位按NumPy2规则保留float32。新增10000单位、np.float32回调的真实账本回归，终值逐值等于显式Python-float ratio原ledger。未修改旧程序。
