# 多主体 SND 启发式共演化

项目包含 SND 实例生成器、多主体博弈求解器、EoH/EoH-S 启发式搜索、结果归档和可视化模块。

### 参考解的自适应计算

训练中的参考解默认采用 `reference_method="auto"`。程序计算综合复杂度
`节点数 + 弧数 + 商品数`（时间展开实例还计入时间点数）：不超过
`reference_complexity_threshold` 时调用 MIP，超过阈值时调用限时参考启发式。
这不是穷举算法，也不会把启发式参考值误称为最优值；每条参考记录会写入
`selected_method` 和 `kind`（`optimal`、`incumbent` 或 `heuristic`）。

在 `examples/multi_agent_demo.py` 的 `DemoConfig` 中可调两个主要开关：

```python
reference_method="auto"                 # auto / heuristic / mip
reference_complexity_threshold=1000      # 综合规模分水岭
reference_heuristic="randomized_residual_dijkstra"
```

大规模实例可选的参考启发式为：
`randomized_residual_dijkstra`（多次随机化残余最短路）、`variable_cost`（变动费用贪心）、
`fixed_cost_greedy`（固定费用感知贪心）和 `capacity_aware`（容量感知贪心）。它们均受
`reference_restarts`、`reference_max_paths`、`reference_timeout` 限制，允许拆分商品，
不遍历所有弧组合。参考结果按实例哈希和完整配置缓存到 `reference_cache_dir`。

## 最终程序的公开实例对照与绘图

运行 `.\.venv\Scripts\python.exe .\final_code_test.py`。测试脚本直接调用 `RUN_DIR/final_solver/solver.py` 中的最终求解器，无需 LLM 或重新训练。要测试其他训练结果，修改脚本顶部的 `RUN_DIR`；`SOLVER_TIMEOUT`、`SOLVER_LINE_LIMIT` 和 `BASELINE_TIMEOUT` 控制测试执行预算。历史训练逻辑保持原版本。

`BENCHMARK_CASES` 包含 32 个公开实例，节点数限制为 10–200，读取 `benchmarks/` 中的 `R.tgz`、`C.tgz`、`CPlus.tgz`、`cexall.zip`、`waxman.tgz`、`planarmnet.tgz`、`gridmnet.tgz`。补充了费用结构、稠密度不同的固定费用 SND，以及 Grid/Planar 线性多商品流实例；不再测试 300/500 节点。缺包时显示原始下载地址，不使用随机实例替代。对照方法包括变动费用贪心、固定费用感知贪心、容量感知贪心、随机多启动构造（内部名称 `grasp`，并非完整的文献 GRASP）。所有成功结果均独立验解、重算费用，失败和超时保留记录。

其中 24 个是固定费用 SND，8 个是零固定费用的线性多商品流特例，分别统计并在图中独立分面。`w1_200_04` 的原始数据违反单节点割容量必要条件，记录证据并跳过，不计作算法失败，也不修改原数据。其余实例通过必要条件审计不代表已证明可行，固定费用 SND 对照最高仍为 150 节点。

每次运行将逐方法更新 `final_comparison.json`、`final_solution.json`，并独立归档到 `benchmarks/results/YYYYMMDD_HHMMSS_ffffff/`。完成后仅生成四张图：逐实例目标值相对差距分组柱状图、各实例运行时间、相对差距热力图、按节点规模的可行解成功率；不同模型类别放在独立子图。每张只输出 PNG（600 dpi）和 PDF，共 8 个图片文件，不生成 SVG 或逐实例图。目标值柱状图以每个实例本次找到的最好可行解为基准，绘制 `100 × (目标值 − 最好值) / |最好值|`，越低越好；最好值为零时分母取 1。零差距用圆点标记，失败、跳过、待执行分别标记 F、S、P，不作为零差距。该差距不是与已知最优解的差距；运行时间比较也应结合报告中各方法不同的执行预算。

图表独立保存到 `benchmark_figures/YYYYMMDD_HHMMSS_ffffff/`；文件名和图内脚注均有时间戳，目录包含所有图片、`plotted_data.json`、`figure_manifest.json` 和 `index.html`，可整体复制。打开 `benchmark_figures/index.html` 可进入最新一批。每次重新绘图也生成新目录，不覆盖历史输出。只重绘已有结果时，将 `PLOT_ONLY = True`；这会读取 `final_comparison.json`，不会重新求解，也不代表一次新的独立实验。

## 静态与时间相关实例生成

`SNDInstance` 默认保持原有 `static-snd-v2` 数据结构。传入 `time_periods=H` 后，生成 `time-expanded-snd-v1`：物理节点复制到时间点 `0..H`；服务弧从 `(i,t)` 到 `(j,t+旅行时间)`，容量和成本按出发时刻随机变化；零费用等待弧从 `(i,t)` 到 `(i,t+1)`；每个商品只在释放时刻注入，在截止时刻交付。固定成本按一次服务出发收取。`time_expansion` 保存物理节点/弧映射、弧时刻表和商品时间窗，顶层 `nodes/arcs/commodities` 为可序列化的时间展开弧流数据。

`time_periods` 是**时间段数**，不是实例数量。例如 `time_periods=3` 表示 `0→1`、`1→2`、`2→3` 三段，因此边界时刻有 `0、1、2、3` 共四个。所有时间段默认等长，但生成器没有规定现实中的单位，也没有存储“每段多少分钟”：使用前应自行确定并记录时间粒度，例如一段 = 1 小时。此时 `time_periods=4` 表示规划未来 4 小时，`max_travel_time=2` 表示单条物理弧的随机旅行时间最多 2 小时；若一段改为 30 分钟，相同参数就分别表示 2 小时和 1 小时。商品的释放时间、截止时间和弧的出发/到达时间也都使用同一“段”单位。时间段越短，时间刻画越细；若要保持相同现实规划时长，就必须相应增大 `time_periods`，展开网络与计算量也会增长。

`snd_solver.SNDSolver`、约束检查、参考值计算、候选评估和最终部署均可直接消费时间展开实例：它们把展开后的节点和弧当作普通弧流网络，因此服务弧的出发/到达顺序、等待和释放/截止时间由拓扑自动约束。静态实例仍走原有路径。多主体 demo 可设置 `DemoConfig(time_periods=4, max_travel_time=2)`；训练范围模式可用 `time_periods_range=(3, 6)`。生成器博弈仅允许扰动展开弧的成本/容量与商品需求，不允许改动时间窗和时刻拓扑；若扰动需求，须同步更新 `time_expansion.commodity_windows[*].demand`。

时间模式的 `path` 效用只统计实际服务弧的跳数，`congestion` 只统计服务弧的容量利用率；等待弧虽承载流量，但不算运输服务。`transport` 指标仍按所有弧的变动费用计算，而等待弧费用为零。静态模式下所有弧均按原规则计入。

```python
from snd_generator import SNDInstance, SNDParameterBounds
from snd_solver import SNDSolver

static = SNDInstance(5, 0.5, 0.7, seed=42)  # 原静态格式
temporal = SNDInstance(5, 0.5, 0.7, seed=42,
                       time_periods=4, max_travel_time=2)
print(temporal.to_dict()["time_expansion"]["commodity_windows"][0])
solution = SNDSolver(temporal, solver_type="scipy", time_limit=30, mip_gap=0.01).solve()
print(solution.objective_value, solution.constraint_satisfaction["feasible"])

bounds = SNDParameterBounds(nodes=(4, 7), time_periods=(3, 5))
sampled = bounds.sample(seed=123)  # 固定 seed 可复现
```

完整示例见 `examples/generate_temporal_instance.py`。求解器仍使用多商品弧流 MIP，但时间窗由展开图拓扑表达，并校验时间映射、弧时刻与商品时间窗。生成器保证每个商品至少有一条时序可达路径，弧容量设置保证这些路径可同时承载所有商品。时间展开会增加网络规模，节点数为 `物理节点数 × (H+1)`；动态网络的 LLM 搜索实验应从小规模开始。

## 直接运行 demo

`examples/multi_agent_demo.py` 使用 `DemoConfig` 直接调用 `MultiAgentSNDSolver`，不再使用 argparse。修改文件末尾 `main()` 中的字段即可运行。

```python
from examples.multi_agent_demo import DemoConfig, run_demo

config = DemoConfig(
    mode="offline",                 # offline / api / replay
    eoh_framework="eoh-s",          # eoh 或 eoh-s
    outer_iterations=2,
    eoh_iterations=1,
    population_size=3,
    n_candidates=1,
    instances_per_generator=2,
    parallel_workers=4,              # 评估进程全局并发上限
    response_workers=1,              # 主体/LLM 并发，接口支持后再增加
    nodes=4,
    coverage=0.5,
    fixed_cost=0.7,
)
result = run_demo(config)
```

要运行时间相关多商品实验，可在同一个 `DemoConfig` 中设置 `time_periods=4, max_travel_time=2`；若希望训练和测试都采样不同时间跨度，改用 `time_periods_range=(3, 6)`。保留 `time_periods=None` 且不设置时间范围时，仍运行静态 SND。

在线模式需要在 `llmapi.env` 中配置 `LLM_MODEL`、`LLM_API_KEY` 和 `LLM_BASE_URL`：

```python
config = DemoConfig(mode="api", model="provider/model")
```

重放历史调用：

```python
config = DemoConfig(mode="replay", replay_file="runs/<run>/llm_calls.jsonl")
```

## 主要参数

| 参数 | 用途 |
|---|---|
| `outer_iterations` | 外层多主体博弈迭代次数 |
| `eoh_iterations` | 每轮启发式搜索迭代次数 |
| `population_size` | 候选种群或 EoH-S 集合大小 |
| `n_candidates` | 每次 LLM 请求生成的候选数量 |
| `instances_per_generator` | 每个生成器的训练实例数量 |
| `validation_instance_count` | 独立验证实例数量；用于逐轮诊断、最终候选门槛及验证 Gap 部署排序，0 表示关闭 |
| `parallel_workers` | 评估线程数及候选、生成器、参考解子进程的全局并发上限，1 为串行 |
| `response_workers` | 主体响应搜索并发数，默认 1；接口支持后可先试 2，最多同时有这么多主体发起 LLM 请求 |
| `generator_operators` | 生成器每轮演化算子，默认全部 `E1/E2/M1/M2/M3`；删减会减少探索预算，求解器不受此项影响 |
| `eoh_framework` | `eoh` 使用原有 EoH，`eoh-s` 使用 CPI/CPM/CS/LS |
| `response_mode` | `pure` 保留原扩池行为；`mpsro` 将多个近似最佳纯程序组成一个混合响应入池 |
| `mixture_support_size` | M-PSRO 每个混合响应最多包含的纯程序数 `m_p` |
| `mixture_epsilon` | M-PSRO 成员效用相对本轮最佳效用的最大允许差 `ε` |
| `eohs_mixture_selection` | EoH-S + M-PSRO 求解器支持集：`cpm` 默认按互补选择顺序保留，`utility` 使用效用容差；生成器始终使用效用容差 |
| `cpi_failure_margin` | CPI 中失败候选相对同实例最差可行 Gap 的正惩罚增量 |
| `eoh_selection` | 原有 EoH 的 `topk` 或 `cluster` 筛选方式 |
| `cluster_method` | 原有 EoH 的文本聚类方法 |
| `cpi_mode` | EoH-S 的 CPI 聚合方式：`mean` 或 `weighted` |
| `strategy_pool_limit` | 每个主体在元博弈中最多保留的策略数；限制联合动作数 |
| `external_deviation_tolerance` | 外部响应偏离总量的提前停止阈值；`None` 关闭 |
| `early_stop_patience` | 外部偏离连续达标的轮数 |
| `cs_enabled` / `ls_enabled` | 是否启用 EoH-S 的互补感知搜索和局部搜索 |
| `generator_directions` | 实例生成方向，如 `cost`、`path`、`congestion` |
| `solver_weights` | 求解器综合效用中各指标的权重 |
| `nodes`、`coverage`、`fixed_cost` | 基础 SND 实例参数 |
| `nodes_range`、`coverage_range`、`fixed_cost_range` | 训练和测试实例的采样范围 |
| `time_periods` | 时间段数；`None` 为静态，正整数启用时间展开 |
| `time_periods_range` | 时间段数采样范围；单独设置也可启用范围采样 |
| `max_travel_time` | 单条物理弧旅行时间上限，单位为时间段 |
| `execution_line_limit` | 候选程序硬执行行数上限；与工作量效用预算独立 |
| `execution_line_limit_scale` | 可选规模系数；按节点、弧、商品和时间点数动态计算上限 |
| `max_code_lines` / `max_code_chars` / `max_ast_nodes` | 候选源码的静态上限，与运行时执行行数上限不同 |
| `stagnation_patience` / `stagnation_restarts` | EoH-S CPI 停滞后的种群重启阈值与最大次数 |
| `minimum_mixture_success` | 混合响应在基础/验证实例上的最小逐成员成功概率 |
| `deployment_selection` | `cce` 按有限池边际选部署候选；`validation_gap` 按验证集 Gap 选 |

## 训练提速与阶段计时

下界计算已由逐商品反复扫弧改为按起点分组的 Dijkstra。同一实例的路径与运输费用下界
按**完整实例内容哈希**在本次运行内缓存；不同求解程序共用结果。实例生成器演化后，
只要容量、费用、需求或其他实例字段发生变化，就使用新键计算，不会误用旧实例的下界。
参考解、生成样本、程序-实例评估还增加了同键并发去重：多个线程同时请求时，只算一次。
参考解仍有原有磁盘缓存，下界缓存不跨运行持久化。

CPU 评估和 LLM 并发是两个独立设置，在 demo 末尾 `main()` 中调整：

```python
parallel_workers=6,        # 按 CPU/内存设置，参考求解也受此全局上限约束
response_workers=1,        # 默认不并发请求；确认服务端配额后可先试 2
generator_operators=("E1", "E2", "M1", "M2", "M3"),  # 默认保留全部探索
```

开启主体并发后，各玩家仍针对同一份旧策略池/均衡快照搜索，全部完成后才更新策略池。
归档写入受锁保护，程序 ID 与各主体的随机序列不依赖线程完成顺序；嵌套评估不再重复创建
多层线程池。并发模式与串行模式的程序 ID/种子规则不同，**不保证两者搜索轨迹一致**；
回放必须沿用原并发设置及其他配置。LLM 并发不减少请求总数，过高可能因限流和重试变慢。

每个新运行目录增加 `timings.jsonl`（逐调用/阶段耗时）和 `timing_summary.json`
（累计阶段耗时、失败数和缓存命中/等待/计算次数），并在 `result.json` 内保存汇总。
其中包括参考解准备、LLM、各主体响应、候选/生成器执行、下界、收益表、CCE、验证选择、
留出测试和绘图。**阶段可能嵌套或并发，不能把各阶段秒数相加当总耗时**；
总墙钟仍看 `total_wall_time`。每轮刷新汇总，异常前已写入的事件可在 JSONL 中查看。

本次提速不自动降低 T/R、训练/验证/测试样本数，不放宽可行性与执行预算，
也不改变禁止暖启动作为最终输出的规则。若主动缩减 `generator_operators`，应另行比较训练质量。
测量本机下界计算的前后差异（不调用 LLM/MIP，也不是全流程训练加速比）：

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_training_performance.py --instance runs\<run>\base_instance.json
```

跨规模采样示例：

```python
config = DemoConfig(
    nodes_range=(4, 10),
    coverage_range=(0.3, 0.8),
    fixed_cost_range=(0.4, 0.9),
    instances_per_generator=5,
    test_instance_count=10,
)
```

## 输出

每次运行会在 `runs/multi_<timestamp>_<id>/` 下生成 `result.json`、`demo_config.json`、程序归档、遗传关系、最终求解器、可视化图片和 LLM 调用日志。训练实例进入搜索和收益矩阵；验证实例用于逐轮诊断及最终部署选择；留出测试集只在搜索结束后评估一次，不反馈到候选搜索或部署选择。`candidate_tests.jsonl` 也标注 `evaluation_split="validation"`；`test_summary` 报告成功率、成功率 Wilson 95% 区间和 Gap，`test_scale_summary` 按物理节点数分层。样本间并非严格独立同分布时，区间只能作描述性参考。

EoH-S 模式会把基础训练网络与当前生成器产生的合法难例一起用于 CPI/CPM；`mean` 对独特实例等权，`weighted` 为活跃生成器难例赋予 `1 + 元策略边际概率` 的权重。纯 EoH-S 集合中的成员全部进入一个复合求解器策略，收益矩阵和部署会运行每个成员、独立验解并选择最低成本的可行解；M-PSRO 模式则按混合权重抽样，并在失败时回退。工作量指标统计实际运行成员的执行行数；部署类返回选中的成员和失败成员。每个成员的超时单独计算，故集合总耗时最多随成员数增长。`work_budget` 只用于工作量效用归一化，`execution_line_limit` 是候选程序的硬执行上限，二者可以独立设置；失败记录还会区分 `line_limit`、`timeout` 和其他运行失败。

暖启动限制为硬规则，无需 demo 开关：非暖启动候选按源码去重后达到 `population_size` 时，EoH/EoH-S 本地种群永久淘汰暖启动，并记录 `warm_start_retired` 事件。即使种群尚未满额，最终输出也只能由非暖启动程序组成；改标签或仅修改注释/排版的暖启动复制品同样被排除。这不证明候选一定优于暖启动，也不判定任意两个不同程序在语义上是否等价；没有通过基础/验证实例部署门槛的候选就拒绝导出。训练归档及初始检查点仍保留暖启动，便于追溯。

### M-PSRO 式混合响应

在 `DemoConfig` 设置 `response_mode="mpsro"`、`mixture_support_size=3` 即可开启。原 EoH 求解器和各生成器按效用容差 `mixture_epsilon` 取近似最佳成员；EoH-S 求解器默认采用 `eohs_mixture_selection="cpm"`，按 CPM 的互补选择顺序保留完整 CPM 集合，因而不受 `mixture_support_size` 截断；该参数主要约束原 EoH/生成器的效用筛选。可切换成 `utility` 恢复统一的效用容差筛选。成员等权混合，不对实例或解的数值字段求平均。收益矩阵对成员效用加权求期望；CCE 是受限多方一般和博弈上的联合分布。

最终 `final_solver/solver.py` 每次根据固定 `seed`、实例内容及策略哈希确定性抽取**一个**非暖启动程序，执行并校验后输出一份完整方案；若抽中的程序失败，会按权重降序尝试其余成员，全部失败才报错。暖启动只用于初始化和训练诊断，绝不进入最终集合/混合支持集或导出文件。训练配对收益仍按原始成员权重计算期望，失败计入惩罚，**不会用部署回退结果替换训练收益**。`members.json` 保存成员 ID、内部权重、seed 和策略哈希。`result.json` 的 `deployment_kind` 为 `mixed`，并保存本轮混合响应的谱系及权重；它的单实例目标值是实际抽中/回退成员的结果，收益矩阵则是混合响应的期望效用，二者不应混淆。M-PSRO 最终代表策略优先遵循求解器玩家的 CCE 边际概率，并在基础实例不可行时记录 `fallback_from_cce`。若没有非暖启动候选，训练抛出明确异常并写出 `selection_status.json`，不生成伪最终程序。

推荐显式分离工作量效用预算与硬执行上限：

```python
utility = UtilityConfig(work_budget=500_000, execution_line_limit=2_000_000,
                        execution_line_limit_scale=5_000)
config = DemoConfig(response_mode="mpsro", eoh_framework="eoh-s",
                    eohs_mixture_selection="cpm", population_size=4,
                    mixture_support_size=4, instances_per_generator=10,
                    validation_instance_count=10, test_instance_count=20,
    execution_line_limit=2_000_000,
                    execution_line_limit_scale=5_000)
```

`final_solver/meta_solver.py` 另提供 `FinalMetaSNDSolver().solve(instance)`：先剔除暖启动及含暖启动的响应，再将保留的 CCE **边际**概率归一化，按此部署分布抽取历史响应及其内部成员。若保留策略的原始概率全为 0，则只部署已通过门槛的最终候选，明确标注 `selected_non_warm_start_fallback_zero_cce_mass`，不冒称原 CCE 边际。相应概率、原始概率、删除的概率质量和规则保存在 `meta_strategy.json`；此部署分布不能重建原多主体 CCE 的相关联合抽样，完整原始概率仍在 `result.json -> meta` 中。

`result.json` 的 `selected_solver_distribution` 列出基础实例上各成员的权重、可行性和 Gap；迭代记录中的 `validation_mixture_distribution` 是验证诊断，`test_mixture_distribution` 保留为 `null`，防止误以为每轮都可看测试集。最终 `test_evaluations`、`test_summary` 与 `test_scale_summary` 是一次性留出报告。若任一成员失败，无条件期望 Gap 记为 `null`。这与固定种子下**实际抽中的一次方案**分开报告。导出目录的 `runtime_config.json` 保存训练时使用的单成员超时和行数上限，导出类默认沿用这些预算；手动覆盖导出类构造参数仍可能改变行为。

EoH-S 记录原始逐实例 Gap，失败先标记为不可用；在同一批候选上，统一转换为有限惩罚：行数超限为“最差可行 Gap + `cpi_failure_margin`”，超时再增加一档惩罚，其他运行失败再增加一档。若该实例所有候选都失败，则从 0 开始按失败类型给出有限惩罚。因此 CPI 始终有限，可行程序不会因 Gap 极大而比失败程序更差。报告同时保留成员的可行率，避免单看 CPI 掩盖失败。

该模式借鉴 Zhang 等（2025）M-PSRO 的混合最佳响应扩池机制，**不直接继承原文的收敛定理**：原文针对双人零和近似纳什均衡，本项目是多方一般和 CCE，且 LLM 搜索与有限实例评估并不保证全策略空间内的近似最佳响应。`response_mode="pure"` 保留此前的 EoH 单程序和 EoH-S 集合择优逻辑。`deployment_selection` 只改变最终部署候选的选择，不改变已经求得的 CCE；因此 `cce_gap` 仍只能解释为当前有限策略池上的约束偏离。

`diagnostic_scope` 在结果中标明：CCE gap 只针对当前有限策略池，外部响应只针对有限 LLM 候选，留出测试只是固定种子的有限样本。`CCEMetaSolver` 显式枚举联合动作，超过 `max_profiles` 会报错；规模更大时需使用其他元求解器和更多独立测试种子。旧 `runs/` 不会自动转成新版实验结果。

注意：旧运行的 `line_limit` 是**执行期间触发的 Python 行事件次数**，不是生成源码的长度。评估日志中 360 条 `line_limit` 记录也不一定对应 360 份不同程序；同一程序可在多个实例上失败。新版在 `failure_summary` 同时报告失败的程序-实例对数量和去重后的程序数，并用 `max_code_lines/max_code_chars/max_ast_nodes` 限制源码，仍保留运行时预算。提高执行上限会增加卡住程序的耗时风险，不能保证 Gap 改善。配置、提示词、JSON 和归档均显式使用 UTF-8；在 PowerShell 查看中文请使用 `Get-Content -Encoding UTF8 <文件>`。终端显示乱码不等于文件内容被破坏。

要支持更可信的大规模泛化结论，建议扩大 `nodes_range`、`time_periods_range`，并在**多个独立 seed** 上运行。demo 当前设为 50 个留出实例，单次运行仍不足以证明泛化。小规模参考值可能是 MIP 的 incumbent，大规模默认参考值是限时启发式；因此相对 Gap 不是严格最优性间隙，应结合 `reference_kind`、`selected_method` 和运行时间一起分析。上述改进不会自动把历史运行约 4% 的 Gap 降到某个目标，也不能证明全局均衡或全分布成功率。

```python
from pathlib import Path
import importlib.util

run_dir = Path(result.output_dir)
spec = importlib.util.spec_from_file_location("final_solver", run_dir / "final_solver" / "solver.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
solution = module.FinalSNDSolver(timeout=30).solve(SNDInstance(5, 0.5, 0.7, seed=99))
print(solution["objective"], solution.get("selected_member"))
```

`result.json` 中 `deployment_kind` 为 `ensemble` 时，`selected_solver_set` 是实际部署成员 ID。CPI 仍是逐实例 oracle 指标；部署规则是运行全部成员后按可行目标值选优，并不需要知道参考最优值。CCE gap 只描述当前策略池；外部响应偏离另行记录，两者不可互代。历史运行结果按当时版本生成，不会自动变成新版实验结果。

## 直接使用求解类

```python
from snd_generator import SNDInstance
from multi_agent_snd import MultiAgentSNDSolver
from multi_agent_snd.baselines import OfflineMultiAgentClient

solver = MultiAgentSNDSolver(
    SNDInstance(4, 0.5, 0.7, seed=42),
    llm_client=OfflineMultiAgentClient(),
    eoh_framework="eoh-s",
    parallel_workers=4,
)
result = solver.solve("降低总成本并保持容量约束可行")
print(result.objective, result.output_dir)
```
