# 本轮新拟合 G1 非线性联合 NSDE 参考市场

新增reference_market.py、test_reference_market.py及本报告；按明确追加授权修改generator_export及其测试以接入新nsde。旧predecessor proof仍为failed，没有放宽其接受标准，也没有修改旧source、weights、marker或模拟文件。

## 固定基底与市场重新拟合的区别

每折旧C1.npz只作为本轮冻结的二维价格基底输入，记录完整SHA，核对xi在原W/bounds有限金融域内。复用基底不等于复用旧市场权重。新模块直接调用当前原market.fit_member(rep,calendar,cfg,新目录,2026,0)，**不替换原非线性head，也不调用linear_head**。

AST守卫强制Dense64(tanh)→Dense64(tanh)→Dense7。原作者联合约束likelihood、Adam(3e-4,clipnorm10)、batch128、最多100epochs、训练内前90%后10%和earlystop patience12全部保持当前原函数。fit_member先真实model.fit，再保存/重载的是新目录刚拟合的权重，没有load旧C1目录权重的分支。

原C1/common.npz仅读time/warm/strike/premium四个共同条件；不会加载旧real_xi/real_cov/gbm或oldsingle模拟。新market.prepare_origins重新计算全部40960个起点、暖启动及合约条件，并按原存储dtype cast逐元素验证这四字段一致。不需要旧proof complete，不读旧market.done或旧single。

## 接口和正式产物

`工作区/运行支持/环境/market/python.exe reference_market.py --end 2025`，或`run(end)`；每次实际重新拟合一折一个2026原非线性member，不做额外bootstrap member，也不增加策略注册项。

输出 `OUT/markets/end/nsde/`：

- market_2026.weights.h5：新fit真实保存的权重。
- simulation.npz：spot `(40960,H)`、xi `(40960,H,2)`、共享summary `(40960,H,4)`、time/warm/strike/premium。
- reference_common.npz：四共同条件、同shock生成的gbm_new、gbm_sigma/gbm_log_mu标量、fresh real_dates/real_xi/scale/real_cov/real_mu、return_dates/returns、nll/nll_dates、归一化金融几何W/b/anchor/radius。
- market.json：complete/status、identity.source=当前reference_market.py字节SHA、actual_original_market_source_sha256、factor/episodes/config/rawspot SHA、父文件/上游代码SHA、完整折配置、架构和训练截止、NLL、2/4子步诊断、输出SHA和时间。

NPZ临时写后原子replace，market.json最后才complete。任意错误写failed并抛出，不发布完整市场身份。

fresh real_xi是根据冻结基底、仅训练期scale以及原.999内缩预处理重新算出的观测state，不是读取旧common中的state；real_cov/nll来自新拟合NN。三者都能与冻结基底语义区分。

## 同条件、同冲击与金融核验

保留origin_seed104、shock_seed105；一次生成全预算 `(40960,H−1,4,3)`，按原方式聚合成2子步。完整40960路径用原reflect模拟，另每8条对应路径用同四子步作配对诊断。原有限域约束不仅检查输入、完整2子步和配对4子步，也在保存float32 xi之后再次检查，金融容差仍是1e-7，没有因存储转换放宽。

摘要使用原center/basis和新xi×scale重建20价格格点，再调用共享summary，512行分块限制内存；不生成原质量控制器quality作为新的策略特征。

GBM闭式参数独立重算并与fit_member返回geometry[6:8]严格核对：只用训练截止前、现货calendar相邻、排除除息的合法logspot增量，前90%估计log_mu和sigma，剩余10%是同原市场的训练内验证。gbm_new用这些参数和共同Brownian第0方向、同日期步长生成。log_mu已经是log漂移，生成时不重复减半方差。

## 新 G1 导出适配

generator_export新增`--model nsde`，从新reference的simulation读spot/xi/summary，按共同条件构建同7个f槽位；前仓仍由策略另加为第8槽位。state_sidecar保留新xi和origin IDs。

G1导出要求market.json表明head_widths=[64,64,7]、market_weights_reused=false、reference_market.py当前SHA一致、simulation SHA匹配；拒绝把linear head或旧权重缓存伪装成新G1。它不读取旧single。G0由root的common_inputs读取gbm_new/gbm_sigma生成；本模块未改common_inputs。

## 测试和2025真实执行结果

先写4项reference测试，模块缺失时全部明确失败。实现后fresh4项通过(.085秒)：原Dense64架构守卫、旧common含不可pickle读取的毒性gbm/real_xi/real_cov对象仍能只读四条件、闭式参数排除未来且只取前90%、金融域及四→二同shock。

nsde导出新增回归先因unsupported generator失败，适配后generator_export共6项通过(3.575秒)，验证新nonlinear身份、sidecar及拒绝伪装linear head。

按本任务授权已实际重新拟合并完整模拟**2025一折**，没有启动其他三折或全策略训练：

| 项目 | 结果 |
|---|---:|
| fresh NN fit | 1.9002635秒 |
| 总运行 | 41.357959秒 |
| 训练/验证模拟路径 | 32768 / 8192 |
| 共同时间点 | 65 |
| validation NLL | −6.289027043335751，192条 |
| frozen基底最小slack | 9.576094184e-10 |
| 新完整2子步最小slack | 1.329050346e-10 |
| 配对4子步最小slack | 2.710770174e-9 |
| 保存float32后最小slack | 8.435989927e-11 |
| GBM log_mu / sigma | .04136985308 / .24803220426 |
| 合法训练转移 / 拟合前90% | 1914 / 1722 |
| 最后拟合增量日 | 2022-03-18，训练截止2023 |

time/warm/strike/premium四字段与旧合约共同条件cast逐元素exact。新reference产物已真实存在且market complete=true；尚未把它导出到真实inputs，等待root整代归档pilot inputs后统一重建，未混用新旧代输入。

固定同seed/相同冻结输入的真实重新训练会可重复：本次新weights文件SHA与此前旧档记录的2026权重SHA相同，NLL也相同。这是直接实际model.fit的确定性重现，不代表加载旧weights；代码及毒性旧状态读取测试明确区分了这两者。新模拟仍按本轮当前source重新生成，不把旧缓存的近似重放当exact授权。

运行有既有TensorFlow缺GPU DLL后退CPU和pkg_resources弃用告警；没有安装/改环境。只有这一折新参考市场完成，其余三折和全部策略训练仍由root协调。
