"""在项目 .venv 中启动的 marimo 多主体演示；按钮控制模型 API 调用。"""
import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import sys
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from multi_agent_snd import MultiAgentSNDSolver, OfflineMultiAgentClient
    from multi_agent_snd import LiteLLMClient
    from snd_generator import SNDInstance
    return LiteLLMClient, MultiAgentSNDSolver, OfflineMultiAgentClient, SNDInstance, mo, project_root


@app.cell
def _(mo):
    # 参数变更后需要重新点击按钮；默认离线，不消耗 API 额度。
    mode = mo.ui.dropdown(["Offline", "API"], value="Offline", label="LLM backend")
    directions = mo.ui.multiselect(["cost", "path", "transport", "fixed", "congestion", "work"],
                                  value=["cost", "path", "congestion"], label="Generator players")
    iterations = mo.ui.number(1, 100, value=2, label="Outer iterations T")
    eoh_rounds = mo.ui.number(1, 100, value=1, label="EoH rounds R")
    nodes = mo.ui.number(2, 100, value=4, label="Nodes")
    coverage = mo.ui.slider(0, 1, step=.05, value=.5, label="Arc coverage")
    fixed_cost = mo.ui.slider(0, 1, step=.05, value=.7, label="Fixed cost adjustment")
    guidance = mo.ui.text_area(value="优先共享已开启弧，显式维护剩余容量，兼顾成本、路径与拥堵。", label="本次引导提示词")
    run_button = mo.ui.run_button(label="开始运行（API 模式会调用 env 中配置的模型）")
    mo.vstack([mode, directions, mo.hstack([iterations, eoh_rounds, nodes]), coverage, fixed_cost, guidance, run_button])
    return coverage, directions, eoh_rounds, fixed_cost, guidance, iterations, mode, nodes, run_button


@app.cell
def _(LiteLLMClient, MultiAgentSNDSolver, OfflineMultiAgentClient, SNDInstance, coverage,
      directions, eoh_rounds, fixed_cost, guidance, iterations, mo, mode, nodes, project_root, run_button):
    mo.stop(not run_button.value, mo.md("请设置参数并点击开始运行。默认离线，不消耗 API 额度。"))
    client = (LiteLLMClient(seed=42, timeout=60, env_file=project_root / "llmapi.env")
              if mode.value == "API" else OfflineMultiAgentClient())
    solver = MultiAgentSNDSolver(
        SNDInstance(int(nodes.value), coverage.value, fixed_cost.value, seed=42),
        llm_client=client, generator_directions=tuple(directions.value),
        outer_iterations=int(iterations.value), eoh_iterations=int(eoh_rounds.value),
        output_dir=project_root / "runs", seed=42)
    result = solver.solve(user_prompt=guidance.value)
    mo.md(f"目标值：**{result.objective}**\n\n全部输出：`{result.output_dir}`")
    return (result,)


@app.cell
def _(mo, result):
    # 展示图片独立于训练 cell；全尺寸 SVG 保存在输出目录。
    mo.vstack([mo.md("### 多方均衡与留出测试"),
               mo.image(str(result.output_dir / "figures" / "equilibrium_diagnostics.png")),
               mo.image(str(result.output_dir / "figures" / "test_gap.png")),
               mo.md("### 程序遗传关系"),
               mo.image(str(result.output_dir / "figures" / "lineage.png"))])
    return


if __name__ == "__main__":
    app.run()
