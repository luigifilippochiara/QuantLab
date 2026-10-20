"""Safety scans of the source tree: no dynamic code execution, LLM calls only where allowed."""

import re
from pathlib import Path

from quantlab.config import PROJECT_ROOT

SRC = PROJECT_ROOT / "src" / "quantlab"

# Built-in calls that run or load code from strings, plus dynamic attribute and
# module access. The look-behind skips methods such as re.compile(...).
FORBIDDEN = re.compile(r"(?<![\w.])(eval|exec|compile|__import__|getattr)\s*\(|import_module\s*\(")


def source_files() -> list[Path]:
    files = sorted(SRC.rglob("*.py"))
    assert files, "no source files found"
    return files


def test_no_eval_exec_or_dynamic_imports_in_src() -> None:
    offenders = []
    for path in source_files():
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if FORBIDDEN.search(line):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{number}: {line.strip()}")
    assert offenders == [], "\n".join(offenders)


def test_scan_would_catch_violations() -> None:
    for bad in ["eval(x)", "exec (code)", "getattr(np, name)", "importlib.import_module(m)",
                "__import__('os')", "compile(src, 'f', 'exec')"]:  # fmt: skip
        assert FORBIDDEN.search(bad), bad
    for fine in ["re.compile(r'x')", "execute_plan(plan)", "evaluation.run()", "model.eval_x"]:
        assert not FORBIDDEN.search(fine), fine


def test_llm_is_used_only_by_planner_and_narrator() -> None:
    """Only planner.py imports the LLM SDKs; only planner.py and narrator.py call the APIs."""
    for sdk in ("import anthropic", "import openai"):
        importers = {p.name for p in source_files() if sdk in p.read_text()}
        assert importers == {"planner.py"}, sdk
    callers = {
        p.name
        for p in source_files()
        if "messages.create(" in p.read_text() or "responses.create(" in p.read_text()
    }
    assert callers == {"planner.py", "narrator.py"}


def test_langgraph_is_used_only_by_the_graph() -> None:
    """LangGraph is the orchestrator's framework only: the graph (graph.py) and its nodes
    (nodes.py) import it, no other module and no page of the app does (the app goes
    through quantlab.graph and quantlab.nodes)."""
    app_files = sorted((PROJECT_ROOT / "app").rglob("*.py"))
    importers = {
        p.name
        for p in [*source_files(), *app_files]
        if re.search(r"^from langgraph|^import langgraph", p.read_text(), re.MULTILINE)
    }
    assert importers == {"graph.py", "nodes.py"}
