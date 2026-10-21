"""多主体 SND 演示：直接实例化求解类。

修改下方 DemoConfig 的字段，然后运行本文件或调用 main() 即可。
本示例不依赖 argparse，适合编辑器、Notebook 和 Marimo 直接运行。
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from snd_generator import SNDInstance, SNDParameterBounds
from multi_agent_snd import CCEMetaSolver, MultiAgentSNDSolver, OfflineMultiAgentClient, UtilityConfig
from multi_agent_snd.storage import write_json


@dataclass
class DemoConfig:
    """一次实验的全部参数；直接修改这个对象即可调整实验。"""

    # LLM 调用模式：offline 不联网，api 从 env_file 调用外部模型，
    # replay 读取 replay_file 中保存的历史响应，适合复现实验。
    mode: Literal["offline", "api", "replay"] = "offline"
    # 提示词：告诉 LLM 本轮希望生成什么方向的 SND 启发式；文本越具体，
    # 生成结果通常越稳定。离线模式下该字段仍会写入实验配置。
    prompt: str = "降低 SND 总成本，同时保持容量约束可行"
    # 外部模型名称；None 表示由环境文件中的模型配置或客户端默认值决定。
    model: str | None = None
    # 环境变量文件路径，通常包含 API key、base URL 和模型名称；仅 api 模式使用。
    env_file: Path = field(default_factory=lambda: PROJECT_ROOT / "llmapi.env")
    # replay 模式的历史调用日志路径；mode 不是 replay 时可保持为 None。
    replay_file: Path | None = None

    # 外层多主体博弈迭代次数；增大可让生成器与求解器相互适应更多轮，
    # 但会近似线性增加 LLM 调用和评估时间。
    outer_iterations: int = 2
    # 每轮内部 EoH/EoH-S 搜索次数；EoH-S 中用于更新互补启发式集合。
    eoh_iterations: int = 1
    # 本地候选池上限 K；较大有利于保留多样策略，但会增加内存和评估开销。
    population_size: int = 3
    # 每次 LLM 请求生成的候选程序数量 n；增大可提高探索范围，也会增加调用成本。
    n_candidates: int = 1
    # 每个实例生成器用于评估候选程序的训练实例数 M；越大评估更稳健但耗时更长。
    instances_per_generator: int = 1
    # LLM 采样温度；较低更确定，较高更有多样性，通常建议 0.2 到 1.2。
    temperature: float = 1.0
    # 全局随机种子：控制实例生成、候选抽样和离线客户端，使结果可复现。
    seed: int = 42
    # 评估线程数及候选/生成器/参考解子进程的全局并发上限；过大可能争抢 CPU、内存。
    parallel_workers: int = 4
    # 主体响应搜索并发：1 保持串行；仅接口支持并发时改为 2–4，独立于 CPU 评估并发。
    response_workers: int = 1
    # 生成器每轮演化算子；默认全部保留。调试可减为 ("E1", "M1", "M2")，会减少探索预算。
    generator_operators: tuple[str, ...] = ("E1", "E2", "M1", "M2", "M3")

    # 搜索框架：eoh 设计单个高性能启发式，eoh-s 设计由互补启发式组成的集合。
    eoh_framework: Literal["eoh", "eoh-s"] = "eoh-s"

    # 策略池扩展方式：pure 保留原行为；mpsro 把多个近似最佳纯程序
    # 等概率组成一个混合响应，再作为一个元博弈动作入池。
    response_mode: Literal["pure", "mpsro"] = "pure"
    # M-PSRO 每个混合响应最多包含多少个纯程序；1 等价于单成员响应。
    mixture_support_size: int = 3
    # M-PSRO 容差：仅纳入效用不低于本轮最佳效用减去此值的程序。
    # 对生成器以及 eohs_mixture_selection="utility" 的求解器生效。
    mixture_epsilon: float = 0.05
    # EoH-S + M-PSRO 求解器混合支持的规则：cpm 保留完整互补代表集合，
    # utility 使用上面的效用容差；support_size 主要约束 utility/原 EoH 响应。
    eohs_mixture_selection: Literal["cpm", "utility"] = "cpm"

    # 以下三个参数仅在 eoh_framework="eoh" 时生效：
    # 候选保留规则，可选 topk（按效果直接取前 K 个）或 cluster（先聚类再选代表）。
    eoh_selection: Literal["topk", "cluster"] = "topk"
    # cluster 模式的聚类算法：kmeans、agglomerative 或 keyword。
    cluster_method: Literal["kmeans", "agglomerative", "keyword"] = "kmeans"
    # cluster 模式允许的最小簇数；实际簇数会结合候选数量自动截断。
    min_clusters: int = 2

    # 以下参数仅在 eoh_framework="eoh-s" 时生效：
    # CPI 的聚合方式：mean 使用实例性能均值，weighted 使用实例权重聚合。
    cpi_mode: Literal["mean", "weighted"] = "mean"
    # CPI 中某实例求解失败时，使用该实例最差可行 Gap 加此正数作惩罚；
    # 避免所有候选失败时 CPI 变为无穷大。
    cpi_failure_margin: float = 1.0
    # 元博弈每个主体最多保留的策略数；较小值可避免 CCE 联合动作数爆炸。
    strategy_pool_limit: int = 8
    # 外部新响应的总收益偏离阈值；连续达标后可提前停止，None 表示关闭。
    external_deviation_tolerance: float | None = 1e-3
    # 需要连续多少轮外部偏离达标才提前停止，避免偶然单轮误判。
    early_stop_patience: int = 2
    # EoH-S CPI 连续无改善达到该轮数后，下一轮重启本地种群。
    stagnation_patience: int = 2
    # EoH-S 最多重启几次；0 表示关闭。
    stagnation_restarts: int = 1
    # 基础与独立验证实例上，混合响应逐成员的最低成功概率；0 关闭门槛。
    minimum_mixture_success: float = 0.0
    # 最终部署策略如何选：cce 遵循有限池边际；validation_gap 用验证集平均 Gap。
    deployment_selection: Literal["cce", "validation_gap"] = "cce"
    # 是否启用互补感知搜索（CS）；关闭后不生成针对集合短板的候选。
    cs_enabled: bool = True
    # 是否启用局部搜索（LS）；关闭后不围绕现有启发式做局部改进。
    ls_enabled: bool = True

    # 实例生成器关注的方向；每个方向对应一个生成器主体和一种效用函数。
    generator_directions: tuple[str, ...] = ("cost", "path", "congestion")
    # 求解器效用函数的权重字典；键名由 UtilityConfig 支持，数值越大影响越强。
    solver_weights: dict[str, float] = field(default_factory=lambda: {
        "cost": 0.55, "path": 0.10, "transport": 0.10,
        "fixed": 0.10, "congestion": 0.10, "work": 0.05,
    })
    # 工作量效用的参考预算（通常按程序规模或执行工作量计）；用于归一化惩罚。
    work_budget: int = 200000
    # 候选程序的硬执行行数上限。与 work_budget 独立；None 使用兼容默认值 work_budget*10。
    execution_line_limit: int | None = 2000000
    # 可选的规模缩放系数；设置后按实例节点/弧/商品/时间点数动态收紧上限，None 为固定上限。
    execution_line_limit_scale: int | None = None
    # LLM 源码质量门槛：分别限制源代码行数、字符数和 AST 节点数；None 表示不启用。
    # 这不是运行时执行行数；运行时仍由 execution_line_limit 防止死循环。
    max_code_lines: int | None = 240
    max_code_chars: int | None = 24000
    max_ast_nodes: int | None = 2000
    # 实例相对基础参数的偏离惩罚权重；越大越抑制生成器产生极端实例。
    distortion_penalty: float = 0.10
    # 候选执行失败时的效用惩罚；越大越倾向于淘汰不稳定程序。
    failure_penalty: float = 2.0

    # 基础 SND 实例的节点数；未设置 nodes_range 时所有训练/测试实例使用该规模。
    nodes: int = 4
    # 基础弧覆盖率，取值通常在 [0, 1]；越大表示网络越稠密。
    coverage: float = 0.5
    # 固定成本调节参数，取值在 [0, 1]；越大通常意味着开启弧的固定成本越高。
    fixed_cost: float = 0.7
    # 节点数整数区间（闭区间）；设置后每个实例会在区间内随机采样节点数。
    nodes_range: tuple[int, int] | None = None
    # 覆盖率浮点区间；设置后覆盖率在该区间内采样，单值 coverage 仅作默认值。
    coverage_range: tuple[float, float] | None = None
    # 固定成本浮点区间；设置后固定成本在该区间内采样，单值 fixed_cost 仅作默认值。
    fixed_cost_range: tuple[float, float] | None = None
    # 时间相关实例：None 表示静态 SND；正整数表示时间段数 H（时间点为 0..H）。
    time_periods: int | None = None
    # 在闭区间内采样 H；单独设置此项也会启用范围模式，无需同时设置 nodes_range。
    time_periods_range: tuple[int, int] | None = None
    # 单条物理弧的最大旅行时间，单位是等长时间段；仅时间模式使用。
    max_travel_time: int = 2
    # 每个生成器额外生成并用于测试的实例数量；越大越能检验泛化能力但耗时越长。
    test_instance_count: int = 3
    # 独立验证集数量；0 表示不启用。验证集只参与最终候选门槛，不反馈训练/博弈。
    validation_instance_count: int = 0

    # 下游 SND 求解器后端；scipy/ortools 通常无需商业许可证，gurobi/cplex 需本地安装。
    solver_type: Literal["scipy", "gurobi", "cplex", "ortools"] = "scipy"
    # 单个 SND 实例的最大求解时间（秒）；超时后返回当前最好可行解。
    time_limit: float = 30.0
    # 混合整数求解相对间隙；0 表示要求最严格，增大可缩短求解时间。
    mip_gap: float = 0.0
    # 参考解选择：auto 按 nodes+arcs+commodities 选择 MIP/启发式。
    reference_method: Literal["auto", "heuristic", "mip"] = "auto"
    reference_complexity_threshold: int = 1000
    # 大规模实例使用的快速参考启发式；不枚举弧组合。
    reference_heuristic: Literal[
        "randomized_residual_dijkstra", "variable_cost", "fixed_cost_greedy", "capacity_aware"
    ] = "randomized_residual_dijkstra"
    reference_restarts: int = 4
    reference_max_paths: int = 16
    reference_timeout: float = 15.0
    reference_cache_dir: Path | None = field(
        default_factory=lambda: PROJECT_ROOT / "runs" / "reference_cache")
    # 单个候选启发式执行的最大时间（秒）；超时候选按失败处理。
    candidate_timeout: float = 30.0
    # 单次 LLM 请求的最大等待时间（秒）；仅 api 模式使用。
    llm_timeout: float = 600.0
    # LLM 调用失败后的重试次数；增大可提高成功率但会延长等待时间。
    llm_retries: int = 2
    # 元策略求解器保存的最大策略画像数量；越大可保留更多历史信息但占用更多内存。
    max_profiles: int = 50000
    # 是否输出底层求解器和评估日志。
    log_output: bool = False
    # 是否在控制台显示“程序启动”和每轮进度。
    progress_output: bool = True
    # 是否生成性能曲线、收益矩阵等图片；关闭可减少运行时间和文件输出。
    make_plots: bool = True
    # 实验输出目录；其中会保存 JSON、候选程序、解释文本和 figures 子目录。
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "runs")


def build_bounds(config: DemoConfig) -> SNDParameterBounds | None:
    """根据范围字段创建实例参数范围。"""
    if (config.nodes_range is None and config.coverage_range is None
            and config.fixed_cost_range is None and config.time_periods_range is None):
        return None
    return SNDParameterBounds(
        nodes=config.nodes_range or (config.nodes, config.nodes),
        coverage=config.coverage_range or (config.coverage, config.coverage),
        fixed_cost=config.fixed_cost_range or (config.fixed_cost, config.fixed_cost),
        time_periods=(config.time_periods_range if config.time_periods_range is not None
                      else ((config.time_periods, config.time_periods)
                            if config.time_periods is not None else None)),
        max_travel_time=config.max_travel_time,
    )


def build_client(config: DemoConfig):
    """根据 mode 创建 LLM 客户端。"""
    if config.mode == "offline":
        return OfflineMultiAgentClient()
    if config.mode == "api":
        from multi_agent_snd import LiteLLMClient
        return LiteLLMClient(model=config.model, seed=config.seed,
                             timeout=config.llm_timeout, env_file=config.env_file)
    if config.mode == "replay":
        if config.replay_file is None:
            raise ValueError("replay 模式必须设置 replay_file")
        from multi_agent_snd.replay import ReplayClient
        return ReplayClient(config.replay_file)
    raise ValueError("mode 必须是 offline、api 或 replay")


def run_demo(config: DemoConfig | None = None):
    """按配置运行一次实验并返回 MultiAgentResult。"""
    config = config or DemoConfig()
    bounds = build_bounds(config)
    base_instance = (bounds.sample(seed=config.seed) if bounds else
                     SNDInstance(config.nodes, config.coverage, config.fixed_cost,
                                 seed=config.seed, time_periods=config.time_periods,
                                 max_travel_time=config.max_travel_time))
    utility = UtilityConfig(
        solver_weights=dict(config.solver_weights), work_budget=config.work_budget,
        execution_line_limit=config.execution_line_limit,
        execution_line_limit_scale=config.execution_line_limit_scale,
        max_code_lines=config.max_code_lines,
        max_code_chars=config.max_code_chars,
        max_ast_nodes=config.max_ast_nodes,
        distortion_penalty=config.distortion_penalty, failure_penalty=config.failure_penalty,
    )
    solver = MultiAgentSNDSolver(
        base_instance, llm_client=build_client(config),
        generator_directions=config.generator_directions,
        outer_iterations=config.outer_iterations, eoh_iterations=config.eoh_iterations,
        population_size=config.population_size, n_candidates=config.n_candidates,
        instances_per_generator=config.instances_per_generator, seed=config.seed,
        temperature=config.temperature, candidate_timeout=config.candidate_timeout,
        llm_timeout=config.llm_timeout, llm_retries=config.llm_retries,
        solver_type=config.solver_type, time_limit=config.time_limit, mip_gap=config.mip_gap,
        log_output=config.log_output, meta_solver=CCEMetaSolver(config.max_profiles),
        utility_config=utility, eoh_selection=config.eoh_selection,
        cluster_method=config.cluster_method, min_clusters=config.min_clusters,
        eoh_framework=config.eoh_framework, cpi_mode=config.cpi_mode,
        cs_enabled=config.cs_enabled, ls_enabled=config.ls_enabled,
        strategy_pool_limit=config.strategy_pool_limit,
        external_deviation_tolerance=config.external_deviation_tolerance,
        early_stop_patience=config.early_stop_patience,
        stagnation_patience=config.stagnation_patience,
        stagnation_restarts=config.stagnation_restarts,
        minimum_mixture_success=config.minimum_mixture_success,
        deployment_selection=config.deployment_selection,
        reference_method=config.reference_method,
        reference_complexity_threshold=config.reference_complexity_threshold,
        reference_heuristic=config.reference_heuristic,
        reference_restarts=config.reference_restarts,
        reference_max_paths=config.reference_max_paths,
        reference_timeout=config.reference_timeout,
        reference_cache_dir=config.reference_cache_dir,
        response_mode=config.response_mode,
        mixture_support_size=config.mixture_support_size,
        mixture_epsilon=config.mixture_epsilon,
        eohs_mixture_selection=config.eohs_mixture_selection,
        cpi_failure_margin=config.cpi_failure_margin,
        parallel_workers=config.parallel_workers, instance_bounds=bounds,
        response_workers=config.response_workers, generator_operators=config.generator_operators,
        test_instance_count=config.test_instance_count,
        validation_instance_count=config.validation_instance_count,
        output_dir=config.output_dir,
        progress_output=config.progress_output, make_plots=config.make_plots,
    )
    print(f"程序启动：mode={config.mode}, framework={config.eoh_framework}", flush=True)
    result = solver.solve(config.prompt)
    demo_config = asdict(config)
    for key in ("env_file", "output_dir", "replay_file", "reference_cache_dir"):
        value = demo_config.get(key)
        demo_config[key] = str(value) if value is not None else None
    write_json(result.output_dir / "demo_config.json", demo_config)
    print(f"运行完成，结果目录：{result.output_dir}")
    print(f"结果 JSON：{result.output_dir / 'result.json'}")
    print(f"最终程序：{result.output_dir / 'final_solver' / 'solver.py'}")
    if config.make_plots:
        print(f"可视化目录：{result.output_dir / 'figures'}")
    return result


def main() -> None:
    """编辑器入口；直接修改下方 DemoConfig 即可运行不同规模的实验。

    当前示例是“8 小时预算下的均衡性优先”配置：保留多轮外层博弈、EoH-S
    的 CS/LS 和 M-PSRO 混合响应，同时限制候选数量、策略池和实例规模，避免
    CCE 联合动作数及参考 MIP 求解时间爆炸。首次运行建议先把
    ``nodes_range``、``time_periods_range`` 和迭代次数再缩小，确认环境、API
    和求解器均正常后再逐步放大。8 小时是运行预算而非严格保证，实际耗时还取决于
    API 延迟、网络密度和求解器难度。所有参数都会写入运行目录的
    ``demo_config.json``，因此结果可以追溯和复现。
    """
    config = DemoConfig(
        # ---------- LLM 与可复现性 ----------
        mode="api",             # "api" 调用环境文件中的模型；"offline" 不联网；"replay" 重放历史调用。
        model="openai/DeepSeek-V4-Flash-Vision-Exp",  # 模型标识；None 表示使用 env 文件中的默认模型。
        seed=42,                 # 全局随机种子，控制实例、候选抽样、混合策略和离线客户端。
        temperature=0.7,         # LLM 采样温度；适度随机以保持多样性，避免过高导致候选不稳定。
        llm_timeout=600.0,        # 单次 LLM 请求最长等待秒数；限制异常请求占用整个实验。
        llm_retries=2,            # 失败后的重试次数；2 次兼顾成功率和总时长。

        # ---------- 搜索框架与多主体博弈 ----------
        eoh_framework="eoh-s",   # "eoh" 设计单个启发式；"eoh-s" 用 CPI/CPM 设计互补集合。
        response_mode="mpsro",   # "pure" 扩展纯策略；"mpsro" 把多个纯程序组成混合响应加入策略池。
        outer_iterations=5,       # 外层博弈轮数 T；3 轮能观察策略互相适应，少于 2 轮不建议用于均衡分析。
        eoh_iterations=2,         # 每轮内部搜索轮数 R；同时启用 CS/LS，避免只靠单轮候选。
        population_size=4,        # 本地候选种群上限 K；4 个成员可提供多样性且评估量可控。
        n_candidates=1,            # 每次请求返回 1 个候选；降低 API 次数，用 EoH 轮数补偿探索。

        # ---------- M-PSRO / EoH-S 超参数 ----------
        mixture_support_size=3,   # utility 模式下每个混合响应最多的纯程序数；CPM 模式保留完整互补集合。
        mixture_epsilon=0.05,     # 允许加入混合的效用差；越大支持集更丰富，越小更偏向高效程序。
        eohs_mixture_selection="cpm",  # "cpm" 补短板；"utility" 按效用容差筛选。
        cpi_mode="weighted",     # CPI 聚合："mean" 等权；"weighted" 提高活跃生成器难例权重。
        cpi_failure_margin=1.0,   # 失败候选相对该实例最差可行 Gap 的惩罚增量，必须为正数。
        cs_enabled=True,          # EoH-S 互补感知搜索：选择行为差异大的父代，补齐实例短板。
        ls_enabled=True,          # EoH-S 局部搜索：围绕当前高质量父代做局部改进。

        # ---------- 策略池、CCE 与并行 ----------
        strategy_pool_limit=4,    # 每个主体最多保留 4 个历史策略；4 个主体最多约 4^4=256 个联合画像。
        max_profiles=2000,        # CCE 联合画像硬上限；留出余量但防止策略池组合爆炸。
        parallel_workers=8,       # 独立实例/候选评估线程数；按服务器 CPU 调整，过大可能争抢内存。
        response_workers=4,       # LLM 接口并发能力未确认，默认串行；确认后先试 2。
        generator_operators=("E1", "E2", "M1", "M2", "M3"), # 保留全部探索；不靠减少搜索预算提速。
        external_deviation_tolerance=1e-3,  # 外部响应偏离阈值；达到该值才可能提前停止。
        early_stop_patience=2,    # 连续满足偏离阈值的轮数；防止单轮偶然收敛导致过早停止。
        stagnation_patience=2,    # EoH-S CPI 连续无改善的轮数；达到后触发一次种群重启。
        stagnation_restarts=1,    # 最多重启次数；设为 0 可关闭重启。
        minimum_mixture_success=1.0, # 混合响应在基础+验证集上的最低成员成功概率；1.0 拒绝脆弱混合。
        deployment_selection="validation_gap", # 仅用于最终部署，不改变 CCE 元策略；选验证 Gap 更低者。

        # ---------- 训练、验证和测试实例 ----------
        instances_per_generator=8,   # 每个生成器的训练实例数 M；8 个实例足以初步估计效用且不会拖慢每轮评估。
        validation_instance_count=8, # 独立验证实例数；只用于最终候选门槛，不反馈博弈训练。
        test_instance_count=50,      # 留出测试实例数；只用于最终报告；50 例仍需多 seed 才能支撑泛化结论。
        nodes=4,                   # 未设置 nodes_range 时的固定物理节点数。
        coverage=0.5,              # 未设置 coverage_range 时的固定弧覆盖率，范围通常为 [0,1]。
        fixed_cost=0.7,             # 未设置 fixed_cost_range 时的固定成本调节参数，通常为 [0,1]。
        nodes_range=(15, 40),       # 物理节点数闭区间；先用中等规模保证参考 MIP 可在时限内完成。
        coverage_range=(0.3, 0.8), # 弧覆盖率闭区间；适度稠密，避免极稠密网络拖慢 MIP。
        fixed_cost_range=(0.4, 0.9), # 固定成本闭区间；控制开弧成本相对变动成本的比例。

        # 时间相关多商品 SND：节点会扩展为“物理节点数 × (H+1)。
        time_periods=1,         # 固定时间段数 H；设置后必须将 time_periods_range 设为 None。
        time_periods_range=None, # 时间段数闭区间；控制时间展开规模，后续可单独扩大 H。
        max_travel_time=1,         # 单条物理弧最大旅行时间（时间段）；仅时间模式生效。

        # ---------- 候选执行与求解器预算 ----------
        work_budget=500_000,       # 工作量效用归一化预算，不是程序执行的硬安全上限。
        execution_line_limit=3_000_0000, # 候选单次执行硬上限；足够容纳中等网络，同时避免死循环。
        execution_line_limit_scale=10000, # 按实例规模动态收紧上限；需与节点范围同步调整。
        max_code_lines=800,          # LLM 源码静态行数上限；超限候选直接拒绝并记录 code_limit。
        max_code_chars=240000,        # LLM 源码字符数上限；避免冗长代码挤占评估预算。
        max_ast_nodes=20000,          # AST 复杂度上限；过滤过度嵌套/重复展开的候选。
        candidate_timeout=45.0,    # 单个候选程序最大运行秒数；超时会进入 CPI 失败惩罚。
        solver_type="scipy",      # 参考解后端；可选 scipy/ortools/gurobi/cplex。
        time_limit=90.0,           # 参考 MIP 单实例最大求解秒数；超时仍使用当前最好可行解。
        mip_gap=0.05,              # 允许 5% 参考间隙；显著减少 MIP 等待，同时保留可用参考值。
        reference_method="auto",  # 按综合复杂度自动选择 MIP 或大规模参考启发式。
        reference_complexity_threshold=500,  # 节点+弧+商品（加时间点）的切换阈值。
        reference_heuristic="randomized_residual_dijkstra",  # 另可选 variable_cost/fixed_cost_greedy/capacity_aware。
        reference_restarts=4,      # 大规模参考启发式的完整构造次数上限。
        reference_max_paths=16,     # 每商品的残余路径增广次数上限。
        reference_timeout=60.0,     # 大规模参考启发式的单实例墙钟预算。
        log_output=False,          # 是否输出底层求解器详细日志；批量实验通常关闭。
        make_plots=True,           # 是否绘图；大规模批量运行可关闭以减少 I/O。
        progress_output=True,      # 是否在终端报告启动、迭代完成和输出路径。
    )
    run_demo(config)


if __name__ == "__main__":
    main()
