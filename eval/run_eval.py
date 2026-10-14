"""Run the planner evaluation and write a Markdown report to eval/reports/.

    uv run python eval/run_eval.py --mode mock     # harness check, no API key needed
    uv run python eval/run_eval.py --mode live     # measures the real LLM planner

Same as `uv run python -m quantlab.cli eval --mode ...`; the logic lives in
quantlab/evaluation.py so the CLI and this script cannot diverge.
"""

import sys

from quantlab.cli import main

if __name__ == "__main__":
    sys.exit(main(["eval", *sys.argv[1:]]))
