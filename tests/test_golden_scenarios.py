"""The example workflows and the mock eval set give exactly the results of their golden records.

tests/fixtures/golden_scenarios.json holds nine records. For each example
workflow, run in mock mode and auto-approved as `cli demo` does, each holds: the
status, every static and runtime issue, the plan, every step's input and output
hashes and metadata, the numbers of every output table and the facts sheet, on
two data sets: the synthetic test data (conftest.py) and the committed FRED
snapshots.

- "before_simplification": examples 1 and 4, recorded at commit 6e1a74b, before
  the capability registry and the guardrails were simplified and their codes
  renumbered (docs/DECISIONS.md).
- "final_examples": examples 2, 3 and 5, which replaced the earlier ones after the
  simplification (commit 08b1547).
- "before_restructure": all five examples, with their narrative's grounding
  check, and the mock eval set (each case's plan, issues and score), recorded at
  commit 4e3e09e, before the guardrails were restructured (action, enforced by,
  category, L01 and N01).
- "before_stage_codes": the same, recorded at commit 81078a9, before the codes
  were named by pipeline stage (R request check, P planner, V validator, E
  executor, N narrator check) and the request check R01 was added.
- "simulated_injection": example 4 and its eval case, which replaced the earlier
  ones after commit a0dc779: a simulated planner follows the prompt injection and
  proposes an unregistered capability, which the validator blocks (V02).
- "safe_haven_flagship": all six examples and the mock eval set, recorded after
  commit fc52c98, when the Swiss franc safe-haven analysis became example 1, the
  earlier flagship moved to example 6 and example 3's forecast request moved to
  the S&P 500.
- "yield_curve_below": all six examples and the mock eval set, recorded after
  commit 24372ea, when flag_regime gained `direction` (below: the lower tail) and
  example 2 asked for the weeks when the yield curve was most negative.
- "vix_flagship": all six examples and the mock eval set, recorded after commit
  1754629, when the flagship flagged stress weeks with the VIX level (example 1) and
  example 5 became the same question with a full-history threshold.
- "buy_the_dip": all six examples and the mock eval set, recorded after commit
  1eecb26, when example 6 became "Buy the dip in tech?" (the bottom 10% of past
  weekly NASDAQ returns, then the following week).

Every result must be the same as recorded. The only allowed differences are the
changes made on purpose since a record was made (MIGRATIONS), applied in order:
the simplification translated guardrail codes and removed two parameters from
the plans; the restructure translated two more codes and renamed an issue's
"severity" to "action" (APPROVAL to APPROVE); the stage naming translated every
code but N01; the simulated injection replaced example 4 and its eval case, which
the older records then no longer vouch for; the renumbering swapped examples 2 and
5, so their eval cases' ids (which start with the example number) were renamed; the
new flagship moved the earlier one, unchanged, to example 6 under a new id (and its
eval case with it). Example 3's new request leaves its recorded result as it was (a
refusal records no dataset). flag_regime's new `direction` parameter adds
direction="above" (the default, the old behaviour) to every recorded flag step, and
example 2's new question replaced it and its eval case; the VIX flagship replaced
examples 1 and 5 and their eval cases; buy the dip replaced example 6 and its eval
case, and the eval case nasdaq_weekly_log_vol, which matched the old example 6 in
mock mode and now matches none; the look-ahead example was removed (with its eval
case), and buy the dip became example 5, its eval case renamed with it. No narrative
raises N01, and no example or eval request raises R01.

The fixture is a frozen reference: a record is never regenerated.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from quantlab.config import PROJECT_ROOT, Settings
from quantlab.data import load_catalog
from quantlab.engine import Session, run_request
from quantlab.evaluation import load_golden, score_case
from quantlab.guardrails import GUARDRAILS, Issue, fill_defaults, validate_plan
from quantlab.mock_scenarios import SCENARIOS
from quantlab.planner import generate_plan
from quantlab.types import DataType, canonical_text

FIXTURE = Path(__file__).parent / "fixtures" / "golden_scenarios.json"

# Simplification: old guardrail code -> new code. Two old codes were merged into
# others (G04's checks into G01, G16's into the new G04 TYPE_MISMATCH); the three
# removed ones (G11, G14, G17) have no new code, and no example workflow raised them.
SIMPLIFICATION_CODES = {
    "G01": "G01", "G02": "G02", "G03": "G03", "G04": "G01", "G05": "G04",
    "G06": "G05", "G07": "G06", "G08": "G07", "G09": "G08", "G10": "G09",
    "G12": "G10", "G13": "G11", "G15": "G12", "G16": "G04",
    "R01": "R01", "R02": "R02", "R03": "R03", "R04": "R04", "R05": "R05",
}  # fmt: skip

# Parameters the simplification removed from the registry: forward returns are
# always one period ahead, and the Newey-West lag is always set by the rule.
REMOVED_PARAMS = {"forward_return": ("horizon",), "hac_mean_difference": ("maxlags",)}

# Restructure: PLANNER_REFUSED became the planner's L01 and FORWARD_DATA_AS_SIGNAL
# took its G11; every other code is unchanged.
RESTRUCTURE_CODES = {
    **{f"G{i:02d}": f"G{i:02d}" for i in range(1, 11)},
    "G11": "L01",
    "G12": "G11",
    **{f"R{i:02d}": f"R{i:02d}" for i in range(1, 6)},
}
# The restructure also renamed an issue's "severity" to "action", with these values.
SEVERITY_TO_ACTION = {"BLOCK": "BLOCK", "APPROVAL": "APPROVE", "WARN": "WARN"}


@dataclass(frozen=True)
class Migration:
    """One deliberate change: how a result recorded before it must look after it.

    Fields:
        codes: old guardrail code -> new code (every code an issue may have).
        removed_params: capability -> parameters removed from its plan steps.
        added_params: capability -> {parameter: value} added to its plan steps (a new
            parameter, at the default that keeps the old behaviour).
        severity_to_action: whether issues' "severity" became "action".
        replaced: example workflows and eval cases replaced on purpose: what an older
            record holds for them no longer applies, and a newer record covers them.
        renamed: example workflows and eval cases renamed on purpose: old id -> new id.
    """

    codes: dict[str, str]
    removed_params: dict[str, tuple[str, ...]]
    severity_to_action: bool
    replaced: tuple[str, ...] = ()
    renamed: dict[str, str] = field(default_factory=dict)
    added_params: dict[str, dict[str, Any]] = field(default_factory=dict)


# Stage naming: L planner -> P, G validator -> V, R executor -> E (same numbers),
# N01 unchanged. The new R01 (the request check) translates no older code.
STAGE_CODES = {
    "L01": "P01",
    **{f"G{i:02d}": f"V{i:02d}" for i in range(1, 12)},
    **{f"R{i:02d}": f"E{i:02d}" for i in range(1, 6)},
    "N01": "N01",
}

# Simulated injection: example 4 (a planner refusal, P01) became a simulated
# planner that follows the injection and is blocked by V02; codes are unchanged.
REPLACED_BY_SIMULATED_INJECTION = ("prompt_injection_block", "s4_prompt_injection")

# Renumbering: examples 2 (look-ahead) and 5 (frequency alignment) swapped places, so
# the eval cases named after their number were renamed; nothing else changed.
SWAPPED_CASE_IDS = {
    "s2_vix_full_sample_lookahead": "s5_vix_full_sample_lookahead",
    "s5_spread_frequency_mismatch": "s2_spread_frequency_mismatch",
}

SIMPLIFICATION = Migration(SIMPLIFICATION_CODES, REMOVED_PARAMS, severity_to_action=False)
RESTRUCTURE = Migration(RESTRUCTURE_CODES, {}, severity_to_action=True)
STAGES = Migration(STAGE_CODES, {}, severity_to_action=False)
INJECTION = Migration(
    {code: code for code in GUARDRAILS}, {}, False, REPLACED_BY_SIMULATED_INJECTION
)
RENUMBERING = Migration({code: code for code in GUARDRAILS}, {}, False, renamed=SWAPPED_CASE_IDS)

# New flagship: the earlier flagship moved, unchanged, to example 6 under a new id,
# and its eval case with it; the Swiss franc analysis took its place as example 1.
MOVED_FLAGSHIP_IDS = {
    "volatility_flag_success": "nasdaq_volatility_success",
    "s1_volatility_flag_success": "s6_nasdaq_volatility",
}
FLAGSHIP_MOVE = Migration(
    {code: code for code in GUARDRAILS}, {}, False, renamed=MOVED_FLAGSHIP_IDS
)

# Direction: flag_regime gained `direction`, whose default ("above") is the old
# behaviour; example 2 now asks for the weeks when the yield curve was most negative,
# so it and its eval case (under both ids it has had) were replaced.
REPLACED_BY_CURVE = (
    "spread_frequency_mismatch_block",
    "s5_spread_frequency_mismatch",
    "s2_spread_frequency_mismatch",
)
DIRECTION = Migration(
    {code: code for code in GUARDRAILS},
    {},
    False,
    replaced=REPLACED_BY_CURVE,
    added_params={"flag_regime": {"direction": "above"}},
)

# VIX flagship: example 1 flags stress weeks with the VIX level instead of NASDAQ
# volatility, and example 5 asks the same question with a full-history threshold, so
# both and their eval cases (under every id they have had) were replaced.
REPLACED_BY_VIX = (
    "chf_safe_haven_success",
    "s1_chf_safe_haven",
    "vix_full_sample_lookahead",
    "s2_vix_full_sample_lookahead",
    "s5_vix_full_sample_lookahead",
)
VIX_FLAGSHIP = Migration({code: code for code in GUARDRAILS}, {}, False, REPLACED_BY_VIX)

# Buy the dip: example 6 (NASDAQ volatility regimes) became a buy-the-dip analysis, so
# it and its eval case (under every id they have had) were replaced, and so was the eval
# case nasdaq_weekly_log_vol, whose mock plan was the old example 6's. The old plan is
# kept as a test fixture (conftest.VOLATILITY_PLAN).
REPLACED_BY_DIP = (
    "volatility_flag_success",
    "nasdaq_volatility_success",
    "s1_volatility_flag_success",
    "s6_nasdaq_volatility",
    "nasdaq_weekly_log_vol",
)
BUY_THE_DIP = Migration({code: code for code in GUARDRAILS}, {}, False, REPLACED_BY_DIP)

# Look-ahead removed: the look-ahead example and its eval case (every id they have had)
# are gone, and nothing replaced them; buy the dip moved from 6 to 5, and its eval case
# was renamed with it.
REMOVED_LOOK_AHEAD = (
    "vix_full_sample_lookahead",
    "s2_vix_full_sample_lookahead",
    "s5_vix_full_sample_lookahead",
)
LOOK_AHEAD_REMOVED = Migration(
    {code: code for code in GUARDRAILS},
    {},
    False,
    replaced=REMOVED_LOOK_AHEAD,
    renamed={"s6_buy_the_dip_tech": "s5_buy_the_dip_tech"},
)

# The changes made since each record was recorded, oldest first.
MIGRATIONS = {
    "before_simplification": (
        SIMPLIFICATION,
        RESTRUCTURE,
        STAGES,
        INJECTION,
        RENUMBERING,
        FLAGSHIP_MOVE,
        DIRECTION,
        VIX_FLAGSHIP,
        BUY_THE_DIP,
        LOOK_AHEAD_REMOVED,
    ),
    "final_examples": (
        RESTRUCTURE,
        STAGES,
        INJECTION,
        RENUMBERING,
        FLAGSHIP_MOVE,
        DIRECTION,
        VIX_FLAGSHIP,
        BUY_THE_DIP,
        LOOK_AHEAD_REMOVED,
    ),
    "before_restructure": (
        RESTRUCTURE,
        STAGES,
        INJECTION,
        RENUMBERING,
        FLAGSHIP_MOVE,
        DIRECTION,
        VIX_FLAGSHIP,
        BUY_THE_DIP,
        LOOK_AHEAD_REMOVED,
    ),
    "before_stage_codes": (
        STAGES,
        INJECTION,
        RENUMBERING,
        FLAGSHIP_MOVE,
        DIRECTION,
        VIX_FLAGSHIP,
        BUY_THE_DIP,
        LOOK_AHEAD_REMOVED,
    ),
    "simulated_injection": (
        RENUMBERING,
        FLAGSHIP_MOVE,
        DIRECTION,
        VIX_FLAGSHIP,
        BUY_THE_DIP,
        LOOK_AHEAD_REMOVED,
    ),
    "safe_haven_flagship": (DIRECTION, VIX_FLAGSHIP, BUY_THE_DIP, LOOK_AHEAD_REMOVED),
    "yield_curve_below": (VIX_FLAGSHIP, BUY_THE_DIP, LOOK_AHEAD_REMOVED),
    "vix_flagship": (BUY_THE_DIP, LOOK_AHEAD_REMOVED),
    "buy_the_dip": (LOOK_AHEAD_REMOVED,),
}
# The records that hold (part of) the mock eval set.
EVAL_RECORDS = (
    "before_restructure",
    "before_stage_codes",
    "simulated_injection",
    "safe_haven_flagship",
    "yield_curve_below",
    "vix_flagship",
    "buy_the_dip",
)

# Output metadata compared step by step (the notes are free text and are not).
META_FIELDS = ("kind", "frequency", "return_method", "periods", "horizon", "source_dataset_ids")


def describe_issue(issue: Issue) -> dict[str, Any]:
    """The parts of an issue that identify it: code, issue id, action and step.

    Who enforces it and its category are read from the GUARDRAILS table by code,
    so they follow from the code (tests/test_guardrails.py checks the table).
    """
    return {
        "code": issue.code,
        "issue_id": issue.issue_id,
        "action": issue.action.value,
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
        significant digits), the facts sheet as displayed, and the narrative's
        grounding check and N01 issues.
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
        "grounding": (
            {"grounded": narrative.grounding.grounded, "ungrounded": narrative.grounding.ungrounded}
            if (narrative := session.narrative)
            else None
        ),
        "narrative_issues": (
            [describe_issue(i) for i in session.narrative.issues] if session.narrative else []
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


def describe_eval_cases(settings: Settings) -> dict[str, dict[str, Any]]:
    """Plan, validate and score every golden eval case in mock mode.

    Args:
        settings: settings naming the data folder (the catalog the plans use).

    Returns:
        Case id -> the plan, its issues and the case's score.
    """
    catalog = load_catalog(settings.catalog_path)
    cases = {}
    for case in load_golden():
        result = generate_plan(case.request, case.datasets, "mock", settings, catalog)
        plan = fill_defaults(result.plan) if result.plan else None
        report = validate_plan(plan, catalog, max_steps=settings.max_steps) if plan else None
        score = score_case(case, settings, catalog, "mock")
        cases[case.id] = {
            "plan": (
                [
                    {"id": s.id, "capability": s.capability, "inputs": s.inputs, "params": s.params}
                    for s in plan.steps
                ]
                if plan
                else None
            ),  # fmt: skip
            "cannot_fulfill": plan.cannot_fulfill if plan else None,
            "issues": [describe_issue(i) for i in (report.issues if report else result.issues)],
            "issue_codes": score.issue_codes,
            "valid": score.valid,
            "refused_or_blocked": score.refused_or_blocked,
            "sequence_match": score.sequence_match,
            "correct": score.correct,
            "guardrails_match": score.guardrails_match,
        }
    return cases


# ---------------------------------------------------------------------------
# The recorded results, migrated to what the current system must produce
# ---------------------------------------------------------------------------


def migrate_issue_id(issue_id: str, migration: Migration) -> str:
    """An issue id with its code translated, e.g. "G11:plan" -> "L01:plan"."""
    old_code, _, rest = issue_id.partition(":")
    return f"{migration.codes[old_code]}:{rest}"


def migrate_issue(issue: dict[str, Any], migration: Migration) -> dict[str, Any]:
    """A recorded issue after one migration: code and issue id translated, action renamed."""
    migrated = {
        **issue,
        "code": migration.codes[issue["code"]],
        "issue_id": migrate_issue_id(issue["issue_id"], migration),
    }
    if migration.severity_to_action:
        migrated["action"] = SEVERITY_TO_ACTION[migrated.pop("severity")]
    return migrated


def migrate_step(step: dict[str, Any], migration: Migration) -> dict[str, Any]:
    """A recorded plan step without the parameters the migration removed, with those it added."""
    removed = migration.removed_params.get(step["capability"], ())
    kept = {k: v for k, v in step["params"].items() if k not in removed}
    return {**step, "params": {**kept, **migration.added_params.get(step["capability"], {})}}


def migrate_result(recorded: dict[str, Any], migration: Migration) -> dict[str, Any]:
    """A recorded workflow result after one migration."""
    return {
        **recorded,
        "static_issues": [migrate_issue(i, migration) for i in recorded["static_issues"]],
        "runtime_issues": [migrate_issue(i, migration) for i in recorded["runtime_issues"]],
        "approvals": [migrate_issue_id(issue_id, migration) for issue_id in recorded["approvals"]],
        "plan": [migrate_step(step, migration) for step in recorded["plan"]],
    }


def migrate_case(recorded: dict[str, Any], migration: Migration) -> dict[str, Any]:
    """A recorded eval case after one migration (its codes re-sorted, as the report sorts them)."""
    plan = recorded["plan"]
    return {
        **recorded,
        "plan": [migrate_step(step, migration) for step in plan] if plan is not None else None,
        "issues": [migrate_issue(i, migration) for i in recorded["issues"]],
        "issue_codes": sorted({migration.codes[code] for code in recorded["issue_codes"]}),
    }


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    """The recorded fixture."""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def replaced_since(record: str) -> set[str]:
    """The example workflows and eval cases replaced since a record was made."""
    return {name for migration in MIGRATIONS[record] for name in migration.replaced}


def expected_results(golden: dict[str, Any], data: str) -> dict[str, list[dict[str, Any]]]:
    """Scenario id -> the results each example workflow must give now (one per record).

    Args:
        golden: the fixture.
        data: "synthetic" or "committed".
    """
    expected: dict[str, list[dict[str, Any]]] = {}
    for record, migrations in MIGRATIONS.items():
        for scenario_id, result in golden[record][data]["scenarios"].items():
            if scenario_id in replaced_since(record):
                continue  # a newer record covers what replaced it
            for migration in migrations:
                result = migrate_result(result, migration)
                scenario_id = migration.renamed.get(scenario_id, scenario_id)
            # N01 did not exist when the older records were made; it must never fire
            # on the example workflows (their template narratives are fully grounded).
            expected.setdefault(scenario_id, []).append({**result, "narrative_issues": []})
    return expected


def compare(actual: dict[str, dict[str, Any]], expected: dict[str, list[dict[str, Any]]]) -> None:
    """Assert each workflow's result equals every expected one, part by part (clear failures)."""
    assert sorted(actual) == sorted(expected)  # every example workflow has a golden record
    for scenario_id, result in actual.items():
        for recorded in expected[scenario_id]:
            for part, value in recorded.items():
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
    for record in MIGRATIONS:
        recorded = golden[record]["committed"]["dataset_sha256"]
        if {d: catalog.entries[d].sha256 for d in recorded} != recorded:
            pytest.skip("the committed FRED snapshots changed since the golden record was made")
    compare(run_scenarios(settings), expected_results(golden, "committed"))


def test_mock_eval_matches_the_golden_record(golden: dict[str, Any], tmp_path: Path) -> None:
    actual = describe_eval_cases(committed_data_settings(tmp_path / "runs"))
    covered = set()
    for record in EVAL_RECORDS:
        recorded = golden[record]["eval"]["cases"]
        for case_id, case in recorded.items():
            if case_id in replaced_since(record):
                continue  # a newer record covers what replaced it
            for migration in MIGRATIONS[record]:
                case = migrate_case(case, migration)
                case_id = migration.renamed.get(case_id, case_id)
            covered.add(case_id)
            for part, value in case.items():
                assert actual[case_id][part] == value, f"{record}: {case_id}: {part} differs"
    assert sorted(actual) == sorted(covered)  # every eval case has a golden record
