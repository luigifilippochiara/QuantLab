"""The five example workflows give exactly the results of their golden record.

tests/fixtures/golden_scenarios.json holds two records. For each example
workflow, run in mock mode and auto-approved as `cli demo` does, each holds: the
status, every static and runtime issue, the plan, every step's input and output
hashes and metadata, the numbers of every output table and the facts sheet, on
two data sets: the synthetic test data (conftest.py) and the committed FRED
snapshots.

- "before_simplification": examples 1 and 4, recorded at commit 6e1a74b, before
  the capability registry and the guardrails were simplified and their codes
  renumbered (docs/DECISIONS.md). They must give the same results. The only
  allowed differences are the ones the simplification made on purpose: guardrail
  codes, and the issue ids that contain them, translated with OLD_TO_NEW_CODES,
  and the parameters it removed from the plans (REMOVED_PARAMS).
- "final_examples": examples 2, 3 and 5, which replaced the earlier ones after the
  simplification (commit 08b1547), recorded with the current codes and compared
  exactly.

The fixture is a frozen reference: a record is never regenerated.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from quantlab.config import PROJECT_ROOT, Settings
from quantlab.data import load_catalog
from quantlab.engine import Session, run_request
from quantlab.guardrails import Issue
from quantlab.mock_scenarios import SCENARIOS
from quantlab.types import DataType, canonical_text

FIXTURE = Path(__file__).parent / "fixtures" / "golden_scenarios.json"

# Old guardrail code -> new code. Two old codes were merged into others (G04's
# checks into G01, G16's into the new G04 TYPE_MISMATCH); the three removed ones
# (G11, G14, G17) have no new code, and no example workflow raised them.
OLD_TO_NEW_CODES = {
    "G01": "G01", "G02": "G02", "G03": "G03", "G04": "G01", "G05": "G04",
    "G06": "G05", "G07": "G06", "G08": "G07", "G09": "G08", "G10": "G09",
    "G12": "G10", "G13": "G11", "G15": "G12", "G16": "G04",
    "R01": "R01", "R02": "R02", "R03": "R03", "R04": "R04", "R05": "R05",
}  # fmt: skip

# Parameters removed from the registry: forward returns are always one period
# ahead, and the Newey-West lag is always set by the rule.
REMOVED_PARAMS = {"forward_return": ("horizon",), "hac_mean_difference": ("maxlags",)}

# Output metadata compared step by step (the notes are free text and are not).
META_FIELDS = ("kind", "frequency", "return_method", "periods", "horizon", "source_dataset_ids")


def describe_issue(issue: Issue) -> dict[str, Any]:
    """The parts of an issue that identify it: code, issue id, severity and step."""
    return {
        "code": issue.code,
        "issue_id": issue.issue_id,
        "severity": issue.severity.value,
        "step_id": issue.step_id,
    }


def describe_session(session: Session) -> dict[str, Any]:
    """Everything a workflow's result is judged on, as plain JSON-ready data.

    Args:
        session: a finished session (see engine.run_request).

    Returns:
        status, static and runtime issues, approvals, the plan that ran (defaults
        filled in), one record per executed step (hashes, sample, metadata), every
        TABLE output in its canonical form (the basis of its hash: floats to 12
        significant digits) and the facts sheet as displayed.
    """
    execution = session.execution
    artifacts = execution.artifacts if execution else {}
    return {
        "status": session.status,
        "static_issues": [describe_issue(i) for i in session.report.issues],
        "runtime_issues": [describe_issue(i) for i in execution.issues] if execution else [],
        "approvals": [approval.issue_id for approval in session.approvals],
        "plan": [
            {"id": s.id, "capability": s.capability, "inputs": s.inputs, "params": s.params}
            for s in session.plan.steps
        ],
        "steps": [
            {
                "step_id": record.step_id,
                "capability": record.capability,
                "status": record.status,
                "input_hashes": record.input_hashes,
                "output_hash": record.output_hash,
                "n_obs": record.n_obs,
                "start": record.start,
                "end": record.end,
                "meta": {name: (record.meta or {}).get(name) for name in META_FIELDS},
            }
            for record in (execution.step_records if execution else [])
        ],
        "tables": {
            step_id: json.loads(canonical_text(artifact))
            for step_id, artifact in artifacts.items()
            if artifact.data_type == DataType.TABLE
        },
        "facts": (
            [[fact.fact_id, fact.display] for fact in session.narrative.facts]
            if session.narrative
            else []
        ),
    }


def run_scenarios(settings: Settings) -> dict[str, dict[str, Any]]:
    """Run the five example workflows in mock mode, auto-approving as `cli demo` does.

    Args:
        settings: settings naming the data folder and an isolated runs folder.

    Returns:
        Scenario id -> describe_session of its run.
    """
    results = {}
    for scenario in SCENARIOS:
        session = run_request(
            scenario.request,
            list(scenario.dataset_ids),
            "mock",
            approve=scenario.auto_approve,
            settings=settings,
        )
        results[scenario.id] = describe_session(session)
    return results


def committed_data_settings(runs_dir: Path) -> Settings:
    """Settings for the committed FRED snapshots, with run records kept out of runs/."""
    return Settings(
        anthropic_api_key=None,
        planner_model="test-model",
        narrator_model="test-model",
        llm_mode="mock",
        data_dir=PROJECT_ROOT / "data",
        runs_dir=runs_dir,
    )


# ---------------------------------------------------------------------------
# The recorded results, translated to what the simplified system must produce
# ---------------------------------------------------------------------------


def translate_issue_id(issue_id: str) -> str:
    """An old issue id with its code translated, e.g. "G12:s3:return_method" -> "G10:s3:..."."""
    old_code, _, rest = issue_id.partition(":")
    return f"{OLD_TO_NEW_CODES[old_code]}:{rest}"


def translate_issue(issue: dict[str, Any]) -> dict[str, Any]:
    """An old issue with its code and its issue id translated."""
    code = OLD_TO_NEW_CODES[issue["code"]]
    return {**issue, "code": code, "issue_id": translate_issue_id(issue["issue_id"])}


def without_removed_params(step: dict[str, Any]) -> dict[str, Any]:
    """A recorded plan step without the parameters the simplification removed."""
    removed = REMOVED_PARAMS.get(step["capability"], ())
    return {**step, "params": {k: v for k, v in step["params"].items() if k not in removed}}


def expected_result(recorded: dict[str, Any]) -> dict[str, Any]:
    """What a workflow must give now: the recorded result, with the allowed changes applied."""
    return {
        **recorded,
        "static_issues": [translate_issue(i) for i in recorded["static_issues"]],
        "runtime_issues": [translate_issue(i) for i in recorded["runtime_issues"]],
        "approvals": [translate_issue_id(issue_id) for issue_id in recorded["approvals"]],
        "plan": [without_removed_params(step) for step in recorded["plan"]],
    }


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    """The recorded fixture."""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def expected_results(golden: dict[str, Any], data: str) -> dict[str, dict[str, Any]]:
    """Scenario id -> the result each example workflow must give now, on one data set.

    Args:
        golden: the fixture.
        data: "synthetic" or "committed".
    """
    before = golden["before_simplification"][data]["scenarios"]
    final = golden["final_examples"][data]["scenarios"]  # current codes: used as recorded
    return {**{sid: expected_result(r) for sid, r in before.items()}, **final}


def compare(actual: dict[str, dict[str, Any]], expected: dict[str, dict[str, Any]]) -> None:
    """Assert each workflow's result equals the expected one, part by part (clear failures)."""
    assert sorted(actual) == sorted(expected)  # every example workflow has a golden record
    for scenario_id, result in actual.items():
        for part, value in expected[scenario_id].items():
            assert result[part] == value, f"{scenario_id}: {part} differs"


def test_scenarios_match_the_golden_record_on_synthetic_data(
    golden: dict[str, Any], settings: Settings
) -> None:
    compare(run_scenarios(settings), expected_results(golden, "synthetic"))


def test_scenarios_match_the_golden_record_on_the_committed_snapshots(
    golden: dict[str, Any], tmp_path: Path
) -> None:
    settings = committed_data_settings(tmp_path / "runs")
    catalog = load_catalog(settings.catalog_path)
    for record in ("before_simplification", "final_examples"):
        recorded = golden[record]["committed"]["dataset_sha256"]
        if {d: catalog.entries[d].sha256 for d in recorded} != recorded:
            pytest.skip("the committed FRED snapshots changed since the golden record was made")
    compare(run_scenarios(settings), expected_results(golden, "committed"))
