"""Documentation stays in sync with the code it describes, and never hardcodes counts."""

import json
import re

from quantlab.config import PROJECT_ROOT
from quantlab.evaluation import load_golden
from quantlab.guardrails import CATEGORY_DEFINITIONS, GUARDRAILS, Category
from quantlab.mock_scenarios import FLAGSHIP_REQUEST, SCENARIOS_BY_ID
from quantlab.registry import REGISTRY

README = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
# The full capability and guardrail tables live in the architecture document; the
# README gives the overview and links to them.
ARCHITECTURE = (PROJECT_ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
GUIDE = (PROJECT_ROOT / "docs" / "GUIDE.md").read_text(encoding="utf-8")


def test_guardrail_table_matches_the_code() -> None:
    for g in GUARDRAILS.values():
        row = (f"| {g.code} | `{g.name}` | {g.action.value} | {g.enforced_by} | "
               f"{g.type.value} | {g.category.value} | {g.description} |")  # fmt: skip
        assert row in ARCHITECTURE, f"docs/ARCHITECTURE.md guardrail row out of date: {g.code}"


def test_docs_name_the_categories_of_the_code() -> None:
    # The category list is the Category enum; the docs name it in the same order, and
    # the architecture document gives each category's definition.
    names = [c.value.lower() for c in Category]
    sentence = f"{', '.join(names[:-1])} or {names[-1]}"
    for name, text in [("README.md", README), ("docs/ARCHITECTURE.md", ARCHITECTURE)]:
        assert sentence in re.sub(r"\s+", " ", text), name
    for category, definition in CATEGORY_DEFINITIONS.items():
        assert f"- **{category.value}**: {definition}" in ARCHITECTURE, category


def test_capability_table_lists_every_capability_with_its_code() -> None:
    for name, spec in REGISTRY.items():
        assert f"| {spec.code} | `{name}` |" in ARCHITECTURE, name


def github_anchor(heading: str) -> str:
    """The anchor GitHub gives a Markdown heading, e.g. "Live mode" -> "live-mode"."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def test_readme_links_and_logo_resolve() -> None:
    # The logo comes first and is the committed SVG, so it renders on GitHub.
    assert README.startswith('<p align="center">\n  <img src="assets/logo.svg"')
    assert (PROJECT_ROOT / "assets" / "logo.svg").read_text(encoding="utf-8").startswith("<svg")
    links = re.findall(r"\]\(([^)#\s]+)(?:#([\w-]+))?\)", README)
    assert links, "the README links to the docs"
    for target, anchor in links:
        if target.startswith("http"):
            continue
        path = PROJECT_ROOT / target
        assert path.exists(), f"README links to a missing file: {target}"
        if anchor:
            headings = re.findall(r"(?m)^#+ (.+)$", path.read_text(encoding="utf-8"))
            assert anchor in {github_anchor(h) for h in headings}, f"{target}#{anchor}"


# A number of capabilities or guardrails ("11 capabilities", "thirteen static
# guardrails") or a range of codes (from one C, E, N, P, R or V code to another) goes stale
# when the registry or the guardrail table changes: counts are computed in the app.
WORDS = "one two three four five six seven eight nine ten eleven twelve twenty"
NUMBER = r"(?:\d+|\w+teen|" + "|".join(WORDS.split()) + ")"  # "\w+teen": thirteen to nineteen
HARDCODED_COUNT = re.compile(
    rf"\b{NUMBER}\s+(?:[\w-]+\s+){{0,2}}?(?:capabilities|guardrails|functions)\b",
    re.IGNORECASE,
)
CODE_RANGE = re.compile(r"\b[CEGLNPRV]\d{2}\s*(?:-|–|—|to|through)\s*[CEGLNPRV]?\d{2}\b")


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
    # resolves its `scenario:` reference to it; the user guide quotes it verbatim.
    assert SCENARIOS_BY_ID["volatility_flag_success"].request is FLAGSHIP_REQUEST
    golden = next(c for c in load_golden() if c.id == "s1_volatility_flag_success")
    assert golden.request == FLAGSHIP_REQUEST
    assert FLAGSHIP_REQUEST in GUIDE  # on one line, character for character
    # The text is written only where it is defined, and quoted in the guide.
    opening = FLAGSHIP_REQUEST[:48]
    texts = {**written_text(), "eval/golden_requests.yaml": (PROJECT_ROOT / "eval" /
             "golden_requests.yaml").read_text(encoding="utf-8")}  # fmt: skip
    assert sorted(path for path, text in texts.items() if opening in text) == [
        "docs/GUIDE.md",
        "src/quantlab/mock_scenarios.py",
    ]


def test_every_notebook_exists() -> None:
    names = sorted(p.name for p in (PROJECT_ROOT / "notebooks").glob("*.ipynb"))
    assert [n[:2] for n in names] == ["01", "02", "03", "04", "05", "06", "07", "08"]
