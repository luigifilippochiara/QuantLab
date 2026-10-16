"""Run records, replay and trace: a run can be saved, reproduced bit-for-bit and explained."""

import hashlib

from quantlab.config import Settings
from quantlab.data import Catalog
from quantlab.executor import execute_plan
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
    trace,
)
from quantlab.monitoring import read_events
from quantlab.registry import registry_version
from quantlab.validation import validate_plan

from conftest import RETURN_METHOD_APPROVAL, SPREAD_PLAN, as_plan, raw_plan

APPROVALS = {RETURN_METHOD_APPROVAL}
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
    assert len(report.steps) == 5 and all(step.match for step in report.steps)
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


def test_scenario_6_trace_from_the_flags_back_to_the_nasdaq_file(
    settings: Settings, catalog: Catalog
) -> None:
    record = run_and_save(SCENARIO_1, settings, catalog)
    result = trace(record, "s5")  # the final flag_regime step
    assert [(s.step_id, s.capability) for s in result.chain] == [
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
