"""Documentation stays in sync with the code it describes, and never hardcodes counts."""

import json
import re

from quantlab.config import PROJECT_ROOT
from quantlab.evaluation import load_golden
from quantlab.guardrails import GUARDRAILS
from quantlab.mock_scenarios import FLAGSHIP_REQUEST, SCENARIOS_BY_ID
from quantlab.registry import REGISTRY

README = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_guardrail_table_matches_the_code() -> None:
    for g in GUARDRAILS.values():
        row = (f"| {g.code} | `{g.name}` | {g.action.value} | {g.enforced_by} | "
               f"{g.category.value} | {g.description} |")  # fmt: skip
        assert row in README, f"README guardrail row out of date: {g.code}"


def test_readme_lists_every_capability_with_its_code() -> None:
    for name, spec in REGISTRY.items():
        assert f"| {spec.code} | `{name}` |" in README, name


# A number of capabilities or guardrails ("11 capabilities", "thirteen static
# guardrails") or a range of codes (from one C, G, L, N or R code to another) goes stale
# when the registry or the guardrail table changes: counts are computed in the app.
WORDS = "one two three four five six seven eight nine ten eleven twelve twenty"
NUMBER = r"(?:\d+|\w+teen|" + "|".join(WORDS.split()) + ")"  # "\w+teen": thirteen to nineteen
HARDCODED_COUNT = re.compile(
    rf"\b{NUMBER}\s+(?:[\w-]+\s+){{0,2}}?(?:capabilities|guardrails|functions)\b",
    re.IGNORECASE,
)
CODE_RANGE = re.compile(r"\b[CGLNR]\d{2}\s*(?:-|–|—|to|through)\s*[CGLNR]?\d{2}\b")


def written_text() -> dict[str, str]:
    """Docs, source, app, tests and notebook cells (not their outputs), by file path."""
    paths = [PROJECT_ROOT / "README.md", PROJECT_ROOT / "CLAUDE.md"]
    paths += sorted((PROJECT_ROOT / "docs").glob("*.md"))
    for folder in ["src", "app", "tests", "scripts"]:
        paths += sorted((PROJECT_ROOT / folder).rglob("*.py"))
    texts = {str(p.relative_to(PROJECT_ROOT)): p.read_text(encoding="utf-8") for p in paths}
    for notebook in sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb")):
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        texts[str(notebook.relative_to(PROJECT_ROOT))] = "\n".join(
            "".join(cell["source"]) for cell in cells
        )
    texts.pop("tests/test_docs.py")  # this file holds the patterns themselves
    return texts


def test_no_hardcoded_counts_or_code_ranges() -> None:
    found = []
    for path, text in written_text().items():
        flat = re.sub(r"\s+", " ", text)  # a phrase may wrap across lines
        found += [f"{path}: {m.group()}" for m in HARDCODED_COUNT.finditer(flat)]
        found += [f"{path}: {m.group()}" for m in CODE_RANGE.finditer(flat)]
    assert found == []


def test_flagship_request_is_one_constant_used_everywhere() -> None:
    # The app, the CLI and the notebooks read the scenario's request; the golden set
    # resolves its `scenario:` reference to it; the README quotes it verbatim.
    assert SCENARIOS_BY_ID["volatility_flag_success"].request is FLAGSHIP_REQUEST
    golden = next(c for c in load_golden() if c.id == "s1_volatility_flag_success")
    assert golden.request == FLAGSHIP_REQUEST
    assert FLAGSHIP_REQUEST in README  # on one line, character for character
    # The text is written only where it is defined, and quoted in the README.
    opening = FLAGSHIP_REQUEST[:48]
    texts = {**written_text(), "eval/golden_requests.yaml": (PROJECT_ROOT / "eval" /
             "golden_requests.yaml").read_text(encoding="utf-8")}  # fmt: skip
    assert sorted(path for path, text in texts.items() if opening in text) == [
        "README.md",
        "src/quantlab/mock_scenarios.py",
    ]


def test_every_notebook_exists() -> None:
    names = sorted(p.name for p in (PROJECT_ROOT / "notebooks").glob("*.ipynb"))
    assert [n[:2] for n in names] == ["01", "02", "03", "04", "05", "06", "07", "08"]
