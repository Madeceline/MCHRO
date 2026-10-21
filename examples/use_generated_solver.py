"""加载某次训练导出的类，求解新的 SND 实例；不访问 LLM。"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    from snd_generator import SNDInstance
    from multi_agent_snd.storage import write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--instance", type=Path, help="可选 SND JSON 文件")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--output", type=Path, default=Path("deployed_solution.json"))
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("final_snd_solver", args.run_dir / "final_solver" / "solver.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = args.instance or SNDInstance(4, .5, .7, seed=args.seed)
    solution = module.FinalSNDSolver(timeout=30).solve(instance)
    write_json(args.output, solution)
    print(f"目标值：{solution['objective']}\n已保存：{args.output.resolve()}")


if __name__ == "__main__":
    main()
