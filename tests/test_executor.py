"""Executor: dispatch whitelist, approval gate, runtime guardrails R01-R05, dry-run agreement."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import build_synthetic_data_dir

from quantlab.data import Catalog, file_sha256, load_catalog
from quantlab.executor import (
    CapabilityRefusedError,
    check_alignment_loss,
    check_low_effective_sample,
    dispatch,
    execute_plan,
)
from quantlab.plan import Plan, Step
from quantlab.registry import REGISTRY
from quantlab.scenarios import SCENARIOS_BY_ID
from quantlab.types import Artifact, ArtifactMeta, DataType, Frequency, SeriesKind
from quantlab.validation import fill_defaults, validate_plan

ALL_APPROVALS = {"G12:s3:return_method", "G10:s5"}


def scenario(scenario_id: str, **step_params: dict) -> Plan:
    """A scenario's canned plan as a Plan, optionally overriding params of some steps."""
    s = SCENARIOS_BY_ID[scenario_id]
    plan = fill_defaults(Plan(plan_id="p", request=s.request, **s.plan))
    for step in plan.steps:
        step.params.update(step_params.get(step.id, {}))
    return plan


def runtime_codes(result) -> set[str]:
    return {i.code for i in result.issues}


# --- dispatch and gates -----------------------------------------------------------


def test_dispatch_refuses_unregistered_names(catalog: Catalog) -> None:
    for name in ["random_forest", "eval", "__import__", "os.system", "load_series "]:
        with pytest.raises(CapabilityRefusedError):
            dispatch(Step(id="s1", capability=name), {}, catalog)


def test_blocked_plan_runs_nothing(catalog: Catalog) -> None:
    result = execute_plan(scenario("C"), catalog, ALL_APPROVALS)
    assert result.status == "blocked"
    assert result.artifacts == {} and result.step_records == []
    assert {"G02", "G08"} <= runtime_codes(result)


def test_unapproved_plan_runs_nothing(catalog: Catalog) -> None:
    result = execute_plan(scenario("A"), catalog, approved_issue_ids=set())
    assert result.status == "blocked"
    assert result.step_records == []
    assert "G12:s3:return_method" in result.error


def test_scenario_a_runs_cleanly_on_synthetic_data(catalog: Catalog) -> None:
    result = execute_plan(scenario("A"), catalog, ALL_APPROVALS)
    assert result.status == "completed", result.error
    assert [r.step_id for r in result.step_records] == [f"s{i}" for i in range(1, 9)]
    assert all(r.output_hash and len(r.output_hash) == 64 for r in result.step_records)
    assert not any(c.startswith("R") for c in runtime_codes(result))
    s7 = result.artifacts["s7"].value
    assert s7["details"]["n_flagged"] >= 20 and s7["details"]["n_normal"] >= 20
    load = result.step_records[0]
    assert load.input_hashes == {"dataset:NASDAQCOM": catalog.entries["NASDAQCOM"].sha256}


def test_execution_is_deterministic(catalog: Catalog) -> None:
    first = execute_plan(scenario("F"), catalog)
    second = execute_plan(scenario("F"), catalog)
    assert [r.output_hash for r in first.step_records] == [
        r.output_hash for r in second.step_records
    ]


@pytest.mark.parametrize("scenario_id", ["A", "B", "F"])
def test_dry_run_metadata_matches_runtime_metadata(scenario_id: str, catalog: Catalog) -> None:
    """The validator's predictions must equal what the capability functions produce."""
    plan = scenario(scenario_id)
    dry_run = validate_plan(plan, catalog).dry_run
    result = execute_plan(plan, catalog, ALL_APPROVALS)
    assert result.status == "completed", result.error
    for step_id, artifact in result.artifacts.items():
        predicted, meta = dry_run[step_id], artifact.meta
        assert predicted["data_type"] == artifact.data_type, step_id
        assert predicted["kind"] == meta.kind, step_id
        assert predicted["frequency"] == meta.frequency, step_id
        assert predicted["return_method"] == meta.return_method, step_id
        assert predicted["horizon"] == meta.horizon, step_id
        assert predicted["periods"] == meta.periods, step_id
        assert predicted["dataset_ids"] == meta.source_dataset_ids, step_id


# --- R01 DATASET_HASH_MISMATCH ----------------------------------------------------


def test_r01_tampered_snapshot_blocks_cleanly(catalog: Catalog) -> None:
    path = catalog.path_for("NASDAQCOM")
    path.write_text(path.read_text() + "2099-01-01,1.0\n")
    result = execute_plan(scenario("A"), catalog, ALL_APPROVALS)
    assert result.status == "blocked"
    assert "R01" in runtime_codes(result)
    assert result.step_records[0].status == "blocked"
    assert result.artifacts == {}


# --- R02 INSUFFICIENT_DATA --------------------------------------------------------


def test_r02_window_too_long_for_the_history(catalog: Catalog) -> None:
    plan = scenario("A", s2={"freq": "M"}, s4={"window": 200})  # 360 months < 3 x 200
    result = execute_plan(plan, catalog, ALL_APPROVALS)
    assert result.status == "blocked"
    assert "R02" in runtime_codes(result)
    assert result.step_records[-1].step_id == "s4"


def test_r02_too_few_flagged_observations(catalog: Catalog) -> None:
    plan = scenario("A", s5={"quantile": 0.99})
    result = execute_plan(plan, catalog, ALL_APPROVALS)
    assert result.status == "blocked"
    r02 = [i for i in result.issues if i.code == "R02"]
    assert r02 and "groups" in r02[0].issue_id


# --- R03 MISSING_DATA -------------------------------------------------------------


def test_r03_gappy_series_warns(tmp_path: Path) -> None:
    build_synthetic_data_dir(tmp_path / "gappy", missing_every=10)  # 10% missing
    gappy = load_catalog(tmp_path / "gappy" / "catalog.json")
    result = execute_plan(scenario("F"), gappy)
    assert result.status == "completed"
    assert "R03" in runtime_codes(result)


def test_r03_silent_on_normal_series(catalog: Catalog) -> None:
    assert "R03" not in runtime_codes(execute_plan(scenario("F"), catalog))


# --- R04 ALIGNMENT_LOSS -----------------------------------------------------------


def _weekly(n: int, start: str, kind: SeriesKind = SeriesKind.RATE_LEVEL) -> Artifact:
    index = pd.date_range(start, periods=n, freq="W-FRI")
    meta = ArtifactMeta(kind, Frequency.W, ["X"], n, None, None)
    return Artifact(pd.Series(np.linspace(1, 2, n), index=index), DataType.SERIES, meta)


def test_r04_alignment_loss() -> None:
    step = Step(id="s5", capability="spread")
    spec = REGISTRY["spread"]
    long, short = _weekly(100, "2020-01-03"), _weekly(50, "2020-12-18")
    issues = check_alignment_loss(step, spec, {"a": long, "b": short})
    assert [i.code for i in issues] == ["R04"]
    assert "'a'" in issues[0].message
    same = _weekly(100, "2020-01-03")
    assert check_alignment_loss(step, spec, {"a": long, "b": same}) == []


def test_r04_real_case_different_start_dates(tmp_path: Path, catalog: Catalog) -> None:
    # Drop the first ~15 years of DGS2, as in the real data (DGS2 starts 14 years after DGS10).
    data = tmp_path / "cut"
    shutil.copytree(catalog.base_dir, data)
    dgs2 = data / "snapshots" / "DGS2.csv"
    lines = dgs2.read_text().splitlines()
    dgs2.write_text("\n".join([lines[0], *lines[4000:]]) + "\n")
    cut = load_catalog(data / "catalog.json")
    # Record the edited file's hash in the catalog, so R01 does not fire first.
    cut.entries["DGS2"] = cut.entries["DGS2"].model_copy(update={"sha256": file_sha256(dgs2)})
    result = execute_plan(scenario("F"), cut)
    assert result.status == "completed"
    assert "R04" in runtime_codes(result)


# --- R05 LOW_EFFECTIVE_SAMPLE -----------------------------------------------------


def _test_result(episodes: int) -> Artifact:
    meta = ArtifactMeta(None, Frequency.W, ["X"], 100, None, None)
    value = {"test": "welch_t", "statistic": 1.0, "p_value": 0.3,
             "details": {"n_flagged_episodes": episodes}}  # fmt: skip
    return Artifact(value, DataType.TEST_RESULT, meta)


def test_r05_few_episodes_warn() -> None:
    step, spec = Step(id="s7", capability="compare_groups"), REGISTRY["compare_groups"]
    assert [i.code for i in check_low_effective_sample(step, spec, _test_result(3))] == ["R05"]
    assert check_low_effective_sample(step, spec, _test_result(12)) == []


def test_failing_capability_stops_the_run_cleanly(catalog: Catalog) -> None:
    # Monthly -> weekly would invent data, so resample raises: the run must stop cleanly.
    plan = Plan(
        plan_id="p",
        request="r",
        dataset_ids=["DGS10"],
        steps=[
            Step(id="s1", capability="load_series", params={"dataset_id": "DGS10"}),
            Step(id="s2", capability="resample", inputs={"series": "s1"}, params={"freq": "M"}),
            Step(id="s3", capability="resample", inputs={"series": "s2"}, params={"freq": "W"}),
        ],
    )
    result = execute_plan(plan, catalog)
    assert result.status == "failed"
    assert "higher frequency" in result.error
    assert [r.status for r in result.step_records] == ["completed", "completed", "failed"]
    assert set(result.artifacts) == {"s1", "s2"}
