"""独立多方博弈可视化：图中文字为英文，无需再次调用 LLM。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch


class MultiAgentVisualizer:
    """接受结果对象、字典或 JSON；同时保存 PNG 和可缩放 SVG。"""

    def __init__(self, result: Any, output_dir: str | Path) -> None:
        if isinstance(result, (str, Path)):
            result = json.loads(Path(result).read_text(encoding="utf-8"))
        elif hasattr(result, "to_dict"):
            result = result.to_dict()
        if result.get("schema") != "multi-agent-snd-v1":
            raise ValueError("需要 multi-agent-snd-v1 结果；旧版零和结果不支持此绘图器")
        self.data, self.root, self.files = result, Path(output_dir), {}
        self.root.mkdir(parents=True, exist_ok=True)

    def _save(self, fig: Any, name: str) -> None:
        fig.tight_layout()
        for extension in ("png", "svg"):
            path = self.root / f"{name}.{extension}"
            # 大谱系的 PNG 限制长边像素；SVG 保留全部可缩放节点，避免内存暴涨。
            dpi = min(130, 12000 / max(fig.get_size_inches()))
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            self.files[f"{name}.{extension}"] = str(path.resolve())
        plt.close(fig)

    def plot_all(self) -> dict[str, str]:
        """均衡诊断、测试、双收益表、种群与全部程序谱系。"""
        history, players = self.data["iterations"], self.data["players"]
        t = [h["iteration"] for h in history]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for i, player in enumerate(players):
            axes[0].plot(t, [h["meta"]["deviation_gains"][i] for h in history], marker="o", label=player)
            axes[1].plot(t[1:], [h["external_deviation_gains"][i] for h in history[1:]], marker="o", label=player)
        axes[0].set(title="Restricted-game CCE deviation", ylabel="Positive deviation gain")
        axes[1].set(title="Empirical external exploitability", ylabel="Oracle gain against previous CCE")
        axes[2].plot(t, [h["meta"]["product_nash_conv"] for h in history], marker="o")
        axes[2].set(title="Product-marginal NashConv", ylabel="Sum of gains (not CCE gap)")
        for ax in axes:
            ax.set_xlabel("Outer iteration")
            ax.grid(alpha=.3)
        axes[0].legend(fontsize=7)
        axes[1].legend(fontsize=7)
        self._save(fig, "equilibrium_diagnostics")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        # 验证集可以用于每轮诊断；测试集只在最终 result 中评估，防止迭代泄漏。
        validation = [h.get("validation_summary", {}) for h in history]
        if any(x.get("count", 0) for x in validation):
            axes[0].plot(t, [x.get("mean_gap_successful") for x in validation], marker="o",
                         label="Validation mean (successful)")
            axes[1].plot(t, [x.get("success_rate") for x in validation], marker="o",
                         label="Validation success")
            axes[0].set(title="Validation relative objective gap",
                        ylabel="(Objective - reference) / |reference|")
            axes[1].set(title="Validation feasibility success", ylabel="Success rate", ylim=(-.05, 1.05))
            axes[0].legend(fontsize=7)
            axes[1].legend(fontsize=7)
        else:
            axes[0].text(.5, .5, "No validation set: test is final-only", ha="center",
                         va="center", transform=axes[0].transAxes)
            axes[1].text(.5, .5, "Final test summary is in result.json", ha="center",
                         va="center", transform=axes[1].transAxes)
            axes[0].set_title("Validation diagnostics disabled")
            axes[1].set_title("Final holdout report")
        for ax in axes:
            ax.set_xlabel("Outer iteration")
            ax.grid(alpha=.3)
        self._save(fig, "test_gap")
        # 最终留出集是搜索结束后一次性计算的，单独画出避免被误读为迭代反馈。
        final_summary = self.data.get("test_summary", {})
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = ["Final success rate", "Final successful Gap"]
        values = [final_summary.get("success_rate") or 0,
                  final_summary.get("mean_gap_successful") or 0]
        ax.bar(labels, values, color=("tab:green", "tab:blue"))
        ax.set_title("Final holdout report (evaluated once after search)")
        ax.grid(axis="y", alpha=.3)
        self._save(fig, "final_holdout")
        last = history[-1]
        for j, direction in enumerate(players[1:]):
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for ax, table, title in zip(axes, (last["solver_tables"][j], last["generator_tables"][j]),
                                         (f"Solver contribution A_{j+1}", f"Generator utility B_{j+1}")):
                matrix = np.asarray(table)
                picture = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
                ax.set(title=f"{direction}: {title}", xlabel="Generator program", ylabel="Solver program")
                ax.set_xticks(range(matrix.shape[1]), last["pools"][direction], rotation=45)
                ax.set_yticks(range(matrix.shape[0]), last["pools"]["solver"])
                if matrix.size <= 100:
                    for (row, col), value in np.ndenumerate(matrix):
                        ax.text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=8)
                fig.colorbar(picture, ax=ax)
            self._save(fig, f"payoff_{direction}")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i, player in enumerate(players):
            axes[0].plot(t, [len(h["pools"][player]) for h in history], marker="o", label=player)
            axes[1].plot(t, [h["meta"]["expected_utilities"][i] for h in history], marker="o", label=player)
        axes[0].set(title="Restricted strategy pools", ylabel="Unique programs")
        axes[1].set(title="Expected utilities under joint CCE", ylabel="Utility")
        for ax in axes:
            ax.set_xlabel("Outer iteration")
            ax.legend(fontsize=7)
            ax.grid(alpha=.3)
        self._save(fig, "population_utilities")
        fig, axes = plt.subplots(len(players), 1, figsize=(9, 2.5 * len(players)), squeeze=False)
        for i, player in enumerate(players):
            axes[i, 0].bar(last["pools"][player], last["meta"]["marginals"][i])
            axes[i, 0].set(title=f"{player}: marginal of joint CCE", ylabel="Probability", ylim=(0, 1.05))
        self._save(fig, "meta_marginals")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        if self.data.get("objective") is not None:
            objective = self.data["objective"]
            reference = self.data.get("reference_value")
            axes[0].bar(["Heuristic", "Reference"] if reference is not None else ["Heuristic"],
                        [objective, reference] if reference is not None else [objective])
            axes[1].bar(["Relative gap"], [self.data.get("gap") or 0])
        else:
            axes[0].text(.5, .5, "No feasible heuristic solution", ha="center", transform=axes[0].transAxes)
            axes[1].text(.5, .5, "Gap unavailable", ha="center", transform=axes[1].transAxes)
        axes[0].set(title="Base-instance objective", ylabel="Total cost")
        axes[1].set(title=f"Reference kind: {self.data.get('reference_kind', 'unknown')}", ylabel="Normalized gap")
        self._save(fig, "objective_gap")
        self._lineage(self.data["programs"], "lineage")
        for player in players:
            self._lineage([p for p in self.data["programs"] if p["player"] == player], f"lineage_{player}")
        return self.files

    def _lineage(self, programs: list[dict], filename: str) -> None:
        """拓扑分层确保父代在左；所有节点保留，含重复代码和淘汰/非法候选。"""
        ids = {p["id"] for p in programs}
        depths, slots, positions = {}, {}, {}
        players = list(dict.fromkeys(p["player"] for p in programs))
        for p in programs:
            depths[p["id"]] = max([depths[parent] + 1 for parent in p["parent_ids"] if parent in ids] + [p["iteration"]])
        max_per_lane = 1
        for p in programs:
            key = (p["player"], depths[p["id"]])
            slots[key] = slots.get(key, 0) + 1
            max_per_lane = max(max_per_lane, slots[key])
        counts = {}
        for p in programs:
            x = depths[p["id"]]
            key = (p["player"], x)
            slot = counts.get(key, 0)
            counts[key] = slot + 1
            positions[p["id"]] = (x, -players.index(p["player"]) * (max_per_lane + 1) - slot)
        fig, ax = plt.subplots(figsize=(max(12, (max(depths.values()) + 1) * 2.3),
                                       max(5, len(players) * (max_per_lane + 1) * .7)))
        colors = plt.get_cmap("tab10")
        selected = self.data["selected_solver"]["id"]
        final_pool = {i for pool in self.data["final_pools"].values() for i in pool}
        for p in programs:
            x, y = positions[p["id"]]
            for parent in p["parent_ids"]:
                if parent in positions:
                    ax.add_patch(FancyArrowPatch(positions[parent], (x, y), arrowstyle="->",
                                                mutation_scale=10, alpha=.4, shrinkA=25, shrinkB=28))
            label = f"{p['id']} | {p['player']}\nT{p['iteration']} / R{p['eoh_iteration']} | {p['operator']}"
            if p["error"]:
                label += "\nREJECTED"
            elif p["id"] == selected:
                label += "\nDEPLOYED"
            elif p["id"] in final_pool:
                label += "\nIN POOL"
            ax.text(x, y, label, ha="center", va="center", fontsize=7,
                    bbox={"boxstyle": "round,pad=.35", "facecolor": colors(players.index(p["player"])),
                          "alpha": .75, "edgecolor": "red" if p["error"] else "black",
                          "linewidth": 3 if p["id"] == selected else 1})
        ax.set_xlim(-.7, max(depths.values()) + .7)
        ys = [p[1] for p in positions.values()]
        ax.set_ylim(min(ys)-1, max(ys)+1)
        ax.set_title(f"Program genealogy ({len(programs)} events)\nT: outer iteration; R: EoH round; arrows: parent to child")
        ax.axis("off")
        self._save(fig, filename)
