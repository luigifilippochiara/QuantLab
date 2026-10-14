"""CLI: demo, run, replay and trace work end to end on synthetic data (mock mode)."""

import pytest

from quantlab.cli import main
from quantlab.config import Settings
from quantlab.lineage import list_runs


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


def test_run_replay_and_trace(cli_env: Settings, capsys: pytest.CaptureFixture) -> None:
    assert main(["run", "F", "--datasets", "DGS10", "DGS2", "--mode", "mock"]) == 0
    run_id = list_runs(cli_env.runs_dir)[0]["run_id"]
    assert main(["replay", run_id]) == 0
    assert "ALL HASHES MATCH" in capsys.readouterr().out
    assert main(["trace", run_id, "s8"]) == 0
    output = capsys.readouterr().out
    assert "s8 adf_test" in output and "DGS10 (FRED DGS10)" in output


def test_run_without_approval_exits_nonzero(cli_env: Settings, capsys) -> None:
    assert main(["run", "A", "--datasets", "NASDAQCOM", "--mode", "mock"]) == 1
    assert "--approve-all" in capsys.readouterr().out
    assert main(["run", "A", "--datasets", "NASDAQCOM", "--mode", "mock", "--approve-all"]) == 0
