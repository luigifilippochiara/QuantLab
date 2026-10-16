"""Executor: dispatch whitelist, approval gate, runtime guardrails R01-R04, dry-run agreement."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.data import Catalog, file_sha256, load_catalog
from quantlab.executor import CapabilityRefusedError, check_alignment_loss, dispatch, execute_plan
from quantlab.plan import Plan, Step
from quantlab.registry import REGISTRY
from quantlab.types import Artifact, ArtifactMeta, DataType, Frequency, SeriesKind
from quantlab.validation import fill_defaults, validate_plan

from conftest import (
    RETURN_METHOD_APPROVAL,
    SARIMA_PLAN,
    SPREAD_PLAN,
    as_plan,
    build_synthetic_data_dir,
    raw_plan,
)

SUCCESS = "volatility_flag_success"  # scenario 1


def with_comparison() -> dict:
    """Scenario 1 plus a correctly aligned compare_groups step (weekly returns vs weekly flags)."""
    raw = raw_plan(SUCCESS)
    raw["steps"].append(
        {"id": "s6", "capability": "compare_groups", "inputs": {"values": "s3", "flags": "s5"}}
    )
    return raw


def runtime_codes(result) -> set[str]:
    return {i.code for i in result.issues}


# --- dispatch and gates -----------------------------------------------------------


def test_dispatch_refuses_unregistered_names(catalog: Catalog) -> None:
    for name in ["sarima", "eval", "__import__", "os.system", "load_series "]:
        with pytest.raises(CapabilityRefusedError):
            dispatch(Step(id="s1", capability=name), {}, catalog)


def test_blocked_plan_runs_nothing(catalog: Catalog) -> None:
    result = execute_plan(as_plan(SARIMA_PLAN), catalog)
    assert result.status == "blocked"
    assert result.artifacts == {} and result.step_records == []
    assert runtime_codes(result) == {"G02"}


def test_unapproved_plan_runs_nothing(catalog: Catalog) -> None:
    result = execute_plan(as_plan(raw_plan(SUCCESS)), catalog, approved_issue_ids=set())
    assert result.status == "blocked"
    assert result.step_records == []
    assert RETURN_METHOD_APPROVAL in result.error


def test_scenario_1_flags_high_volatility_weeks(catalog: Catalog) -> None:
    result = execute_plan(as_plan(raw_plan(SUCCESS)), catalog, {RETURN_METHOD_APPROVAL})
    assert result.status == "completed", result.error
    assert [r.step_id for r in result.step_records] == ["s1", "s2", "s3", "s4", "s5"]
    assert all(r.output_hash and len(r.output_hash) == 64 for r in result.step_records)
    assert not any(c.startswith("R") for c in runtime_codes(result))
    flags = result.artifacts["s5"].value
    # The synthetic index has calm and stressed regimes: some weeks are flagged, not most.
    assert 0.02 < flags.mean() < 0.5
    # Flagging starts after 52 weeks of volatility history (min_periods).
    assert len(flags) == len(result.artifacts["s4"].value) - 52
    load = result.step_records[0]
    assert load.input_hashes == {"dataset:NASDAQCOM": catalog.entries["NASDAQCOM"].sha256}


def test_compare_groups_on_aligned_inputs(catalog: Catalog) -> None:
    result = execute_plan(as_plan(with_comparison()), catalog, {RETURN_METHOD_APPROVAL})
    assert result.status == "completed", result.error
    table = result.artifacts["s6"].value
    returns, flags = result.artifacts["s3"].value, result.artifacts["s5"].value
    common = returns.loc[flags.index]  # flags cover a sub-period of the returns
    assert table["n_flagged"] == int(flags.sum())
    assert table["n_flagged"] + table["n_normal"] == len(flags)
    assert table["mean_flagged"] == pytest.approx(common[flags].mean())
    assert table["diff_means"] == pytest.approx(table["mean_flagged"] - table["mean_normal"])


def test_execution_is_deterministic(catalog: Catalog) -> None:
    first = execute_plan(as_plan(SPREAD_PLAN), catalog)
    second = execute_plan(as_plan(SPREAD_PLAN), catalog)
    assert [r.output_hash for r in first.step_records] == [
        r.output_hash for r in second.step_records
    ]


@pytest.mark.parametrize("name", ["scenario_1", "spread", "comparison"])
def test_dry_run_metadata_matches_runtime_metadata(name: str, catalog: Catalog) -> None:
    """The validator's predictions must equal what the capability functions produce."""
    raw = {"scenario_1": raw_plan(SUCCESS), "spread": SPREAD_PLAN, "comparison": with_comparison()}
    plan = as_plan(raw[name])
    dry_run = validate_plan(plan, catalog).dry_run
    result = execute_plan(plan, catalog, {RETURN_METHOD_APPROVAL})
    assert result.status == "completed", result.error
    for step_id, artifact in result.artifacts.items():
        predicted, meta = dry_run[step_id], artifact.meta
        assert predicted["data_type"] == artifact.data_type, step_id
        assert predicted["kind"] == meta.kind, step_id
        assert predicted["frequency"] == meta.frequency, step_id
        assert predicted["return_method"] == meta.return_method, step_id
        assert predicted["periods"] == meta.periods, step_id
        assert predicted["dataset_ids"] == meta.source_dataset_ids, step_id


# --- R01 DATASET_HASH_MISMATCH ----------------------------------------------------


def test_r01_tampered_snapshot_blocks_cleanly(catalog: Catalog) -> None:
    path = catalog.path_for("NASDAQCOM")
    path.write_text(path.read_text() + "2099-01-01,1.0\n")
    result = execute_plan(as_plan(raw_plan(SUCCESS)), catalog, {RETURN_METHOD_APPROVAL})
    assert result.status == "blocked"
    assert "R01" in runtime_codes(result)
    assert result.step_records[0].status == "blocked"
    assert result.artifacts == {}


# --- R02 INSUFFICIENT_DATA --------------------------------------------------------


def test_r02_window_too_long_for_the_history(catalog: Catalog) -> None:
    # 30 years of monthly closes give 359 monthly returns, fewer than 3 x 200.
    plan = Plan(
        plan_id="p",
        request="r",
        dataset_ids=["NASDAQCOM"],
        steps=[
            Step(id="s1", capability="load_series", params={"dataset_id": "NASDAQCOM"}),
            Step(id="s2", capability="resample", inputs={"series": "s1"}, params={"freq": "M"}),
            Step(id="s3", capability="compute_returns", inputs={"series": "s2"},
                 params={"method": "simple"}),
            Step(id="s4", capability="rolling_volatility", inputs={"series": "s3"},
                 params={"window": 200}),
        ],
    )  # fmt: skip
    result = execute_plan(fill_defaults(plan), catalog, {"G12:s3:return_method"})
    assert result.status == "blocked"
    assert "R02" in runtime_codes(result)
    assert result.step_records[-1].step_id == "s4"


# --- R03 MISSING_DATA -------------------------------------------------------------


def test_r03_gappy_series_warns(tmp_path: Path) -> None:
    build_synthetic_data_dir(tmp_path / "gappy", missing_every=10)  # 10% missing
    gappy = load_catalog(tmp_path / "gappy" / "catalog.json")
    result = execute_plan(as_plan(SPREAD_PLAN), gappy)
    assert result.status == "completed"
    assert "R03" in runtime_codes(result)


def test_r03_silent_on_normal_series(catalog: Catalog) -> None:
    assert "R03" not in runtime_codes(execute_plan(as_plan(SPREAD_PLAN), catalog))


# --- R04 ALIGNMENT_LOSS -----------------------------------------------------------


def _weekly(n: int, start: str, kind: SeriesKind = SeriesKind.RATE_LEVEL) -> Artifact:
    index = pd.date_range(start, periods=n, freq="W-FRI")
    meta = ArtifactMeta(kind, Frequency.W, ["X"], n, None, None)
    return Artifact(pd.Series(np.linspace(1, 2, n), index=index), DataType.SERIES, meta)


def test_r04_alignment_loss() -> None:
    step = Step(id="s3", capability="spread")
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
    result = execute_plan(as_plan(SPREAD_PLAN), cut)
    assert result.status == "completed"
    assert "R04" in runtime_codes(result)


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
