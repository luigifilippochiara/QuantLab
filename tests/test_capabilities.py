"""Numerical correctness of the capability functions on tiny hand-checkable series.

Expected values are computed by hand (or with plain Python math), not with the
same pandas call the function uses, so a wrong formula cannot pass by repeating itself.
"""

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from quantlab import capabilities as cap
from quantlab.data import Catalog
from quantlab.types import Artifact, ArtifactMeta, DataType, Frequency, SeriesKind


def make_artifact(
    values: list[float] | np.ndarray,
    kind: SeriesKind = SeriesKind.PRICE_LEVEL,
    freq: Frequency = Frequency.W,
    start: str = "2024-01-05",
    data_type: DataType = DataType.SERIES,
    **meta: object,
) -> Artifact:
    """Build a weekly (Friday-dated) artifact for tests."""
    pandas_freq = {"D": "B", "W": "W-FRI", "M": "ME"}[freq.value]
    index = pd.date_range(start, periods=len(values), freq=pandas_freq, name="date")
    series = pd.Series(values, index=index)
    base = ArtifactMeta(
        kind=kind,
        frequency=freq,
        source_dataset_ids=["TEST"],
        n_obs=len(series),
        start=None,
        end=None,
    )
    for key, value in meta.items():
        setattr(base, key, value)
    return Artifact(value=series, data_type=data_type, meta=base)


# --- returns, differences, spread ------------------------------------------------


def test_simple_returns() -> None:
    out = cap.compute_returns(make_artifact([100.0, 110.0, 99.0]), method="simple")
    assert out.value.tolist() == pytest.approx([0.10, -0.10])
    assert out.meta.kind == SeriesKind.RETURNS
    assert out.meta.return_method == "simple"
    assert out.meta.n_obs == 2


def test_log_returns() -> None:
    out = cap.compute_returns(make_artifact([100.0, 110.0, 99.0]), method="log")
    assert out.value.tolist() == pytest.approx([math.log(1.1), math.log(0.9)])
    assert out.meta.return_method == "log"


def test_multi_period_returns() -> None:
    out = cap.compute_returns(make_artifact([100.0, 110.0, 99.0, 121.0]), "simple", periods=2)
    # 99/100 - 1 and 121/110 - 1
    assert out.value.tolist() == pytest.approx([-0.01, 0.10])
    assert out.meta.periods == 2


def test_returns_reject_non_positive_levels() -> None:
    with pytest.raises(ValueError, match="positive"):
        cap.compute_returns(make_artifact([1.0, 0.0, 2.0]), method="simple")


def test_difference() -> None:
    out = cap.difference(make_artifact([1.0, 1.5, 1.25], kind=SeriesKind.RATE_LEVEL), periods=1)
    assert out.value.tolist() == pytest.approx([0.5, -0.25])
    assert out.meta.kind == SeriesKind.CHANGES


def test_spread_uses_inner_join() -> None:
    a = make_artifact([5.0, 5.1, 5.2], kind=SeriesKind.RATE_LEVEL, start="2024-01-05")
    b = make_artifact([4.0, 4.2, 4.4], kind=SeriesKind.RATE_LEVEL, start="2024-01-12")
    out = cap.spread(a, b)
    # Common Fridays: Jan 12 (5.1 - 4.0) and Jan 19 (5.2 - 4.2).
    assert out.value.index.strftime("%Y-%m-%d").tolist() == ["2024-01-12", "2024-01-19"]
    assert out.value.tolist() == pytest.approx([1.1, 1.0])
    assert out.meta.kind == SeriesKind.SPREAD


# --- rolling volatility -----------------------------------------------------------


def _sample_std(xs: list[float]) -> float:
    """Sample standard deviation with n - 1 in the denominator, by hand."""
    mean = sum(xs) / len(xs)
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))


def test_rolling_volatility_weekly_annualised() -> None:
    rets = [0.01, -0.02, 0.03, 0.00, 0.01]
    returns = make_artifact(rets, kind=SeriesKind.RETURNS, periods=1)
    out = cap.rolling_volatility(returns, window=4)
    expected = [_sample_std(rets[0:4]) * math.sqrt(52), _sample_std(rets[1:5]) * math.sqrt(52)]
    assert out.value.tolist() == pytest.approx(expected)
    # The first value sits on the date of the 4th return (trailing window).
    assert out.value.index[0] == returns.value.index[3]
    assert out.meta.kind == SeriesKind.VOLATILITY
    assert "sqrt(52)" in out.meta.notes[0]


def test_rolling_volatility_daily_and_unannualised() -> None:
    rets = [0.01, -0.02, 0.03, 0.00]
    daily = make_artifact(rets, kind=SeriesKind.RETURNS, freq=Frequency.D)
    assert cap.rolling_volatility(daily, 4).value.iloc[0] == pytest.approx(
        _sample_std(rets) * math.sqrt(252)
    )
    assert cap.rolling_volatility(daily, 4, annualize=False).value.iloc[0] == pytest.approx(
        _sample_std(rets)
    )


def test_rolling_volatility_of_multi_period_returns_scales_by_periods() -> None:
    rets = [0.01, -0.02, 0.03, 0.00]
    two_week = make_artifact(rets, kind=SeriesKind.RETURNS, periods=2)
    # 2-week returns: 26 per year, not 52.
    assert cap.rolling_volatility(two_week, 4).value.iloc[0] == pytest.approx(
        _sample_std(rets) * math.sqrt(26)
    )


# --- forward returns --------------------------------------------------------------


def test_forward_return_simple_compounds_from_t_plus_1() -> None:
    returns = make_artifact(
        [0.10, -0.10, 0.20, 0.05], kind=SeriesKind.RETURNS, return_method="simple", periods=1
    )
    out = cap.forward_return(returns, horizon=2)
    # t0: (1 - 0.10)(1 + 0.20) - 1 = 0.08 ; t1: (1 + 0.20)(1 + 0.05) - 1 = 0.26
    assert out.value.tolist() == pytest.approx([0.08, 0.26])
    assert out.value.index.tolist() == returns.value.index[:2].tolist()
    assert out.meta.horizon == 2
    assert out.meta.kind == SeriesKind.FORWARD_RETURN


def test_forward_return_log_sums() -> None:
    returns = make_artifact(
        [0.10, -0.10, 0.20, 0.05], kind=SeriesKind.RETURNS, return_method="log", periods=1
    )
    out = cap.forward_return(returns, horizon=2)
    assert out.value.tolist() == pytest.approx([0.10, 0.25])


def test_forward_return_needs_known_one_period_returns() -> None:
    no_method = make_artifact([0.1, 0.2, 0.3], kind=SeriesKind.RETURNS)
    with pytest.raises(ValueError, match="method"):
        cap.forward_return(no_method, horizon=1)
    two_period = make_artifact(
        [0.1, 0.2, 0.3], kind=SeriesKind.RETURNS, return_method="log", periods=2
    )
    with pytest.raises(ValueError, match="one-period"):
        cap.forward_return(two_period, horizon=1)


# --- regime flags -----------------------------------------------------------------


def test_expanding_flags_match_independent_computation() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=80)
    out = cap.flag_regime(
        make_artifact(values, kind=SeriesKind.VOLATILITY), 0.8, "expanding", min_periods=20
    )
    # Independent check: for each t >= 20, compare x_t with the quantile of x_0..x_{t-1}.
    expected = [values[t] > np.quantile(values[:t], 0.8) for t in range(20, 80)]
    assert out.value.tolist() == expected
    assert out.data_type == DataType.FLAGS
    assert out.value.dtype == bool


def test_expanding_flags_have_no_look_ahead() -> None:
    """Changing values after t must not change any flag at or before t."""
    rng = np.random.default_rng(1)
    values = rng.normal(size=120)
    original = cap.flag_regime(make_artifact(values), 0.9, "expanding", min_periods=20).value

    t = 70
    altered_values = values.copy()
    altered_values[t + 1 :] = rng.normal(loc=50.0, size=len(values) - t - 1)  # wild future
    altered = cap.flag_regime(make_artifact(altered_values), 0.9, "expanding", min_periods=20).value

    cutoff = original.index[t - 20]  # flags start at position 20
    pd.testing.assert_series_equal(original[:cutoff], altered[:cutoff])
    assert not original.equals(altered)  # the future did change later flags


def test_full_sample_flags_use_one_threshold() -> None:
    values = np.arange(1.0, 21.0)
    out = cap.flag_regime(make_artifact(values), 0.9, "full_sample", min_periods=20)
    assert out.value.tolist() == (values > np.quantile(values, 0.9)).tolist()
    assert "future" in out.meta.notes[0]


def test_count_episodes() -> None:
    flags = pd.Series([False, True, True, False, True, False, False, True])
    assert cap.count_episodes(flags) == 3
    assert cap.count_episodes(pd.Series([True, True])) == 1
    assert cap.count_episodes(pd.Series([False, False])) == 0


# --- statistics -------------------------------------------------------------------


def test_summary_stats() -> None:
    xs = [1.0, 2.0, 4.0, 8.0, 3.0]
    table = cap.summary_stats(make_artifact(xs)).value
    assert table["count"] == 5
    assert table["mean"] == pytest.approx(3.6)
    assert table["std"] == pytest.approx(_sample_std(xs))
    assert (table["min"], table["max"]) == (1.0, 8.0)
    assert table["skew"] == pytest.approx(stats.skew(xs, bias=False))
    assert table["excess_kurtosis"] == pytest.approx(stats.kurtosis(xs, fisher=True, bias=False))
    assert table["start"] == "2024-01-05"


def _values_and_flags(n: int = 200, horizon: int = 4) -> tuple[Artifact, Artifact]:
    """Forward-return-like values and episodic flags on the same weekly dates."""
    rng = np.random.default_rng(42)
    flag_values = np.zeros(n, dtype=bool)
    for start in range(10, n, 25):  # 8 episodes of 6 weeks
        flag_values[start : start + 6] = True
    values = rng.normal(0.0, 0.02, n) - 0.01 * flag_values
    fwd = make_artifact(
        values, kind=SeriesKind.FORWARD_RETURN, horizon=horizon, return_method="simple"
    )
    flags = make_artifact(flag_values, kind=SeriesKind.FLAGS, data_type=DataType.FLAGS)
    return fwd, flags


def test_compare_groups_matches_scipy() -> None:
    fwd, flags = _values_and_flags()
    out = cap.compare_groups(fwd, flags, test="welch_t")
    x, f = fwd.value.to_numpy(), flags.value.to_numpy()
    direct = stats.ttest_ind(x[f], x[~f], equal_var=False)
    assert out.value["statistic"] == pytest.approx(direct.statistic)
    assert out.value["p_value"] == pytest.approx(direct.pvalue)
    d = out.value["details"]
    assert (d["n_flagged"], d["n_normal"]) == (f.sum(), (~f).sum())
    assert d["diff_means"] == pytest.approx(x[f].mean() - x[~f].mean())
    assert d["n_flagged_episodes"] == 8


def test_compare_groups_mann_whitney_and_nonoverlapping() -> None:
    fwd, flags = _values_and_flags(horizon=4)
    out = cap.compare_groups(fwd, flags, test="mann_whitney", nonoverlapping=True)
    x, f = fwd.value.to_numpy()[::4], flags.value.to_numpy()[::4]
    direct = stats.mannwhitneyu(x[f], x[~f], alternative="two-sided")
    assert out.value["statistic"] == pytest.approx(direct.statistic)
    assert out.value["p_value"] == pytest.approx(direct.pvalue)
    assert out.meta.n_obs == 50


def test_ols_hac_matches_statsmodels_and_defaults_to_horizon_minus_1() -> None:
    fwd, flags = _values_and_flags(horizon=4)
    out = cap.ols_hac(fwd, flags)
    design = sm.add_constant(flags.value.astype(float).rename("flag"))
    direct = sm.OLS(fwd.value, design).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    d = out.value["details"]
    assert d["maxlags"] == 3
    assert d["coef_flag"] == pytest.approx(direct.params["flag"])
    assert d["se_flag"] == pytest.approx(direct.bse["flag"])
    assert out.value["p_value"] == pytest.approx(direct.pvalues["flag"])
    # The dummy coefficient is exactly the difference in group means.
    x, f = fwd.value.to_numpy(), flags.value.to_numpy()
    assert d["coef_flag"] == pytest.approx(x[f].mean() - x[~f].mean())
    assert cap.ols_hac(fwd, flags, maxlags=0).value["details"]["maxlags"] == 0


def test_adf_matches_statsmodels() -> None:
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.normal(size=300)) * 0.1 + rng.normal(size=300)
    out = cap.adf_test(make_artifact(x, kind=SeriesKind.SPREAD), regression="ct")
    direct = adfuller(x, regression="ct", autolag="AIC", result_object=True)
    assert out.value["statistic"] == pytest.approx(direct.statistic)
    assert out.value["p_value"] == pytest.approx(direct.pvalue)
    assert out.value["details"]["used_lags"] == direct.lags
    assert out.value["details"]["critical_value_5pct"] == pytest.approx(
        direct.critical_values["5%"]
    )


# --- resample and load ------------------------------------------------------------


def test_resample_weekly_last_and_mean() -> None:
    # Mon 2024-01-01 .. Fri 2024-01-12: two full business weeks.
    daily = make_artifact(np.arange(1.0, 11.0), freq=Frequency.D, start="2024-01-01")
    last = cap.resample(daily, freq="W", how="last")
    assert last.value.index.strftime("%Y-%m-%d").tolist() == ["2024-01-05", "2024-01-12"]
    assert last.value.tolist() == [5.0, 10.0]
    assert last.meta.frequency == Frequency.W
    assert last.meta.kind == SeriesKind.PRICE_LEVEL
    assert cap.resample(daily, freq="W", how="mean").value.tolist() == [3.0, 8.0]


def test_resample_drops_incomplete_last_period() -> None:
    # Mon 2024-01-01 .. Tue 2024-01-09: the second week is still in progress.
    daily = make_artifact(np.arange(1.0, 8.0), freq=Frequency.D, start="2024-01-01")
    out = cap.resample(daily, freq="W")
    assert out.value.index.strftime("%Y-%m-%d").tolist() == ["2024-01-05"]


def test_resample_refuses_upsampling() -> None:
    monthly = make_artifact([1.0, 2.0, 3.0], freq=Frequency.M, start="2024-01-31")
    with pytest.raises(ValueError, match="higher frequency"):
        cap.resample(monthly, freq="W")


def test_load_series_sets_metadata(catalog: Catalog) -> None:
    out = cap.load_series("DGS10", catalog=catalog)
    entry = catalog.entries["DGS10"]
    assert out.meta.kind == SeriesKind.RATE_LEVEL
    assert out.meta.frequency == Frequency.D
    assert out.meta.n_dropped_missing == entry.n_missing
    assert out.meta.n_obs == entry.n_rows - entry.n_missing
    assert out.meta.source_dataset_ids == ["DGS10"]
