"""Documentation stays in sync with the code it describes."""

from quantlab.config import PROJECT_ROOT
from quantlab.guardrails import GUARDRAILS
from quantlab.registry import REGISTRY

README = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_guardrail_table_matches_the_code() -> None:
    for g in GUARDRAILS.values():
        row = f"| {g.code} | `{g.name}` | {g.severity.value} | {g.stage} | {g.risk} |"
        assert row in README, f"README guardrail row out of date: {g.code}"


def test_readme_lists_every_capability() -> None:
    for name in REGISTRY:
        assert f"`{name}`" in README, name


def test_every_notebook_exists() -> None:
    names = sorted(p.name for p in (PROJECT_ROOT / "notebooks").glob("*.ipynb"))
    assert [n[:2] for n in names] == ["01", "02", "03", "04", "05", "06", "07", "08"]
