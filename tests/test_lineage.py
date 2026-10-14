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
from quantlab.plan import Plan
from quantlab.registry import registry_version
from quantlab.scenarios import SCENARIOS_BY_ID
from quantlab.validation import fill_defaults, validate_plan

APPROVALS = {"G12:s3:return_method"}


def run_and_save(scenario_id: str, settings: Settings, catalog: Catalog) -> RunRecord:
    """Execute a canned scenario plan and save a run record, as the engine does."""
    s = SCENARIOS_BY_ID[scenario_id]
    plan = fill_defaults(Plan(plan_id="p", request=s.request, **s.plan))
    result = execute_plan(plan, catalog, APPROVALS)
    record = RunRecord(
        run_id=new_run_id(),
        created_at="2026-01-01T00:00:00Z",
        status=result.status,
        request=s.request,
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
    record = run_and_save("A", settings, catalog)
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
    record = run_and_save("A", settings, catalog)
    report = replay(record.run_id, settings)
    assert report.all_match, report.message
    assert len(report.steps) == 8 and all(step.match for step in report.steps)
    assert read_events(settings.events_path)[-1]["event_type"] == "replay_completed"


def test_replay_detects_changed_data(settings: Settings, catalog: Catalog) -> None:
    record = run_and_save("F", settings, catalog)
    path = catalog.path_for("DGS2")
    path.write_text(path.read_text().replace(",3.", ",9.", 1))
    report = replay(record.run_id, settings)
    assert not report.all_match
    assert "DGS2" in report.message
    assert [c.match for c in report.datasets] == [True, False]


def test_trace_walks_back_to_the_source_file(settings: Settings, catalog: Catalog) -> None:
    record = run_and_save("F", settings, catalog)
    result = trace(record, "s8")  # adf_test on the spread
    # s8 <- s5 (spread) <- s3, s4 (resample) <- s1, s2 (load); s6/s7 are not ancestors.
    assert [step.step_id for step in result.chain] == ["s8", "s5", "s4", "s3", "s2", "s1"]
    assert {source.dataset_id for source in result.sources} == {"DGS10", "DGS2"}
    dgs10 = next(source for source in result.sources if source.dataset_id == "DGS10")
    assert dgs10.sha256 == catalog.entries["DGS10"].sha256
    assert result.chain[0].library_calls.startswith("statsmodels")


def test_list_runs_newest_first(settings: Settings, catalog: Catalog) -> None:
    first = run_and_save("F", settings, catalog)
    second = run_and_save("A", settings, catalog)
    rows = list_runs(settings.runs_dir)
    assert {row["run_id"] for row in rows} == {first.run_id, second.run_id}
    assert rows[0]["run_id"] >= rows[1]["run_id"]
