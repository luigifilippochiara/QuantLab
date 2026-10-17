"""Run records, replay and trace: a run can be saved, reproduced bit-for-bit and explained."""

import hashlib
import json

from quantlab.config import Settings
from quantlab.data import Catalog
from quantlab.engine import run_request
from quantlab.executor import StepRecord, execute_plan
from quantlab.guardrails import validate_plan
from quantlab.lineage import (
    Approval,
    RunRecord,
    dataset_versions,
    environment_info,
    list_runs,
    load_run,
    new_run_id,
    replay,
    save_run,
    step_capability_code,
    trace,
)
from quantlab.mock_scenarios import CANNED_REQUESTS_BY_ID
from quantlab.monitoring import read_events
from quantlab.registry import REGISTRY, registry_version

from conftest import SCENARIO_1_APPROVALS, SPREAD_PLAN, as_plan, raw_plan

APPROVALS = set(SCENARIO_1_APPROVALS)
SCENARIO_1 = raw_plan("volatility_flag_success")


def run_and_save(raw: dict, settings: Settings, catalog: Catalog) -> RunRecord:
    """Execute a plan dict and save a run record, as the engine does."""
    plan = as_plan(raw)
    result = execute_plan(plan, catalog, APPROVALS)
    record = RunRecord(
        run_id=new_run_id(),
        created_at="2026-01-01T00:00:00Z",
        status=result.status,
        request=plan.request,
        plan_final=plan.model_dump(),
        validation=validate_plan(plan, catalog),
        approvals=[
            Approval(issue_id=i, code=i[:3], step_id="s3", message="m", approved_at="t")
            for i in APPROVALS
        ],
        datasets=dataset_versions(result.step_records, catalog),
        environment=environment_info(),
        registry_version=registry_version(),
        steps=result.step_records,
    )
    save_run(record, result.artifacts, settings.runs_dir)
    return record


def test_saved_outputs_hash_to_the_manifest_hashes(settings: Settings, catalog: Catalog) -> None:
    record = run_and_save(SCENARIO_1, settings, catalog)
    loaded = load_run(record.run_id, settings.runs_dir)
    assert loaded.status == "completed"
    outputs = settings.runs_dir / record.run_id / "outputs"
    for step in loaded.steps:
        files = list(outputs.glob(f"{step.step_id}_*"))
        assert len(files) == 1
        assert hashlib.sha256(files[0].read_bytes()).hexdigest() == step.output_hash
    assert loaded.datasets[0].dataset_id == "NASDAQCOM"
    assert loaded.environment["packages"]["pandas"]


def test_replay_reproduces_every_hash(settings: Settings, catalog: Catalog) -> None:
    record = run_and_save(SCENARIO_1, settings, catalog)
    report = replay(record.run_id, settings)
    assert report.all_match, report.message
    assert len(report.steps) == len(SCENARIO_1["steps"])
    assert all(step.match for step in report.steps)
    assert read_events(settings.events_path)[-1]["event_type"] == "replay_completed"


def test_replay_detects_changed_data(settings: Settings, catalog: Catalog) -> None:
    record = run_and_save(SPREAD_PLAN, settings, catalog)
    path = catalog.path_for("DGS2")
    path.write_text(path.read_text().replace(",3.", ",9.", 1))
    report = replay(record.run_id, settings)
    assert not report.all_match
    assert "DGS2" in report.message
    assert [c.match for c in report.datasets] == [True, False]


def test_trace_walks_back_to_the_source_file(settings: Settings, catalog: Catalog) -> None:
    record = run_and_save(SPREAD_PLAN, settings, catalog)
    result = trace(record, "s3")  # the spread
    # s3 <- s1, s2 (load); s4 (summary statistics) comes later, so it is not an ancestor.
    assert [step.step_id for step in result.chain] == ["s3", "s2", "s1"]
    assert {source.dataset_id for source in result.sources} == {"DGS10", "DGS2"}
    dgs10 = next(source for source in result.sources if source.dataset_id == "DGS10")
    assert dgs10.sha256 == catalog.entries["DGS10"].sha256
    assert result.chain[0].library_calls.startswith("pandas.concat")


def test_list_runs_newest_first(settings: Settings, catalog: Catalog) -> None:
    first = run_and_save(SPREAD_PLAN, settings, catalog)
    second = run_and_save(SCENARIO_1, settings, catalog)
    rows = list_runs(settings.runs_dir)
    assert {row["run_id"] for row in rows} == {first.run_id, second.run_id}
    assert rows[0]["run_id"] >= rows[1]["run_id"]


def test_scenario_6_trace_from_the_test_back_to_the_nasdaq_file(
    settings: Settings, catalog: Catalog
) -> None:
    record = run_and_save(SCENARIO_1, settings, catalog)
    result = trace(record, "s8")  # the final step: the significance test and its p-value
    # Both inputs are traced: the forward returns (s6) and the flags (s5) behind them,
    # down to the one weekly price series (s2) they share. compare_groups (s7) is not
    # an ancestor of the test.
    assert [(s.step_id, s.capability) for s in result.chain] == [
        ("s8", "hac_mean_difference"),
        ("s6", "forward_return"),
        ("s5", "flag_regime"),
        ("s4", "rolling_volatility"),
        ("s3", "compute_returns"),
        ("s2", "resample"),
        ("s1", "load_series"),
    ]
    assert all(s.output_hash and s.library_calls for s in result.chain)
    (source,) = result.sources
    entry = catalog.entries["NASDAQCOM"]
    assert (source.fred_id, source.sha256, source.retrieved_at) == (
        entry.fred_id,
        entry.sha256,
        entry.retrieved_at,
    )
    assert source.start and source.end and source.rows_used


def test_replay_of_a_period_and_extremes_run_matches(settings: Settings) -> None:
    """Both new capabilities replay identically; the manifest keeps the resolved period."""
    canned = CANNED_REQUESTS_BY_ID["nasdaq_volatility_peak_since_2020"]
    session = run_request(canned.request, list(canned.dataset_ids), "mock", approve=True,
                          settings=settings)  # fmt: skip
    assert session.status == "completed"

    record = load_run(session.run_id, settings.runs_dir)
    period = next(step for step in record.steps if step.capability == "filter_period")
    assert period.meta["period_start"] == "2020-01-01"
    assert period.meta["period_end"] == period.end  # the last weekly observation
    extremes = settings.runs_dir / session.run_id / "outputs" / "s6_find_extremes.json"
    assert json.loads(extremes.read_text())["max"]["tie_count"] == 1

    report = replay(session.run_id, settings)
    assert report.all_match, report.message
    assert [s.capability for s in report.steps][-2:] == ["filter_period", "find_extremes"]


def test_manifest_records_each_steps_capability_code(settings: Settings, catalog: Catalog) -> None:
    record = run_and_save(SCENARIO_1, settings, catalog)
    manifest = json.loads((settings.runs_dir / record.run_id / "manifest.json").read_text())
    for step in manifest["steps"]:
        assert step["capability_code"] == REGISTRY[step["capability"]].code
    assert manifest["steps"][3]["capability_code"] == "C06"  # rolling_volatility


def test_manifest_records_the_forward_horizon_and_method(
    settings: Settings, catalog: Catalog
) -> None:
    record = run_and_save(SCENARIO_1, settings, catalog)
    manifest = json.loads((settings.runs_dir / record.run_id / "manifest.json").read_text())
    forward = manifest["steps"][5]
    assert forward["capability"] == "forward_return"
    assert forward["params"] == {"method": "log", "horizon": 1}
    assert (forward["meta"]["horizon"], forward["meta"]["return_method"]) == (1, "log")
    assert manifest["steps"][7]["meta"]["horizon"] == 1  # inherited by the test's table
    test = settings.runs_dir / record.run_id / "outputs" / "s8_hac_mean_difference.json"
    assert {"p_value", "se_hac", "maxlags_used", "n_episodes_flagged"} <= set(
        json.loads(test.read_text())
    )
    assert manifest["environment"]["packages"]["statsmodels"]


def test_manifests_from_before_codes_still_replay_and_trace(
    settings: Settings, catalog: Catalog
) -> None:
    """A run record without codes: replay still matches, and codes come from the names."""
    record = run_and_save(SCENARIO_1, settings, catalog)
    path = settings.runs_dir / record.run_id / "manifest.json"
    manifest = json.loads(path.read_text())
    for step in manifest["steps"]:
        del step["capability_code"]
    path.write_text(json.dumps(manifest))
    old = load_run(record.run_id, settings.runs_dir)
    assert all(step.capability_code is None for step in old.steps)

    report = replay(record.run_id, settings)
    assert report.all_match, report.message
    assert [step.capability_code for step in report.steps] == [
        "C01", "C02", "C03", "C06", "C07", "C12", "C09", "C13"
    ]  # fmt: skip
    chain = trace(old, "s5").chain
    assert [step.capability_code for step in chain] == ["C07", "C06", "C03", "C02", "C01"]


def test_a_retired_capability_has_no_code() -> None:
    step = StepRecord(step_id="s4", capability="ols_hac", library_calls="statsmodels",
                      params={}, input_step_ids={}, input_hashes={})  # fmt: skip
    assert step_capability_code(step) is None
