"""单独重绘某次运行的全部图片，不训练、不调用 LLM。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    from snd_visualization.multi_agent import MultiAgentVisualizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.result.parent / "figures"
    for name, path in MultiAgentVisualizer(args.result, output).plot_all().items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
