"""CLI: demo, run, replay and trace work end to end on synthetic data (mock mode)."""

import pytest

from quantlab.cli import main
from quantlab.config import Settings
from quantlab.lineage import list_runs
from quantlab.narrator import NUMBERS_CHECKED_NOTE


@pytest.fixture
def cli_env(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Point the CLI (which reads settings from the environment) at the synthetic data."""
    monkeypatch.setenv("QUANTLAB_DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("QUANTLAB_RUNS_DIR", str(settings.runs_dir))
    return settings


def test_demo_behaves_as_expected(cli_env: Settings, capsys: pytest.CaptureFixture) -> None:
    assert main(["demo"]) == 0
    output = capsys.readouterr().out
    assert "All scenarios behaved as expected." in output
    assert "MISMATCH" not in output
    assert len(list_runs(cli_env.runs_dir)) == 6
    for scenario_id in ["chf_safe_haven_success", "spread_frequency_mismatch_block",
                        "buy_the_dip_tech"]:  # fmt: skip
        assert scenario_id in output
    # The audit trail: scenario 1's significance test traced down to both files.
    trace = output.split("Audit trail (traceability)")[1]
    for line in ["s8 C10 hac_mean_difference", "s6 C08 forward_return", "s5 C07 flag_regime",
                 "s4 C02 resample", "s3 C02 resample", "s2 C01 load_series", "s1 C01 load_series",
                 "VIXCLS (FRED VIXCLS) sha256", "DEXSZUS (FRED DEXSZUS) sha256"]:  # fmt: skip
        assert line in trace


def test_run_replay_and_trace(cli_env: Settings, capsys: pytest.CaptureFixture) -> None:
    run = ["run", "1", "--datasets", "VIXCLS", "DEXSZUS", "--mode", "mock", "--approve-all"]
    assert main(run) == 0
    run_id = list_runs(cli_env.runs_dir)[0]["run_id"]
    assert f"  Note: {NUMBERS_CHECKED_NOTE}" in capsys.readouterr().out  # under the narrative
    assert main(["replay", run_id]) == 0
    replayed = capsys.readouterr().out
    assert "ALL HASHES MATCH" in replayed and "C10" in replayed
    assert main(["trace", run_id, "s8"]) == 0
    output = capsys.readouterr().out
    assert "s8 C10 hac_mean_difference" in output and "DEXSZUS (FRED DEXSZUS)" in output
    assert "VIXCLS (FRED VIXCLS)" in output


def test_run_without_approval_exits_nonzero(cli_env: Settings, capsys) -> None:
    assert main(["run", "1", "--datasets", "VIXCLS", "DEXSZUS", "--mode", "mock"]) == 1
    assert "--approve-all" in capsys.readouterr().out
