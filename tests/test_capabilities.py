"""Numerical correctness of the capability functions on tiny hand-checkable series.

Expected values are computed by hand (or with plain Python math), not with the
same pandas call the function uses, so a wrong formula cannot pass by repeating itself.
"""

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

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


# --- summary statistics -----------------------------------------------------------


def _adjusted_skew_and_kurtosis(xs: list[float]) -> tuple[float, float]:
    """Bias-adjusted sample skewness (G1) and excess kurtosis (G2), from their textbook formulas."""
    n = len(xs)
    mean = sum(xs) / n
    m2, m3, m4 = (sum((x - mean) ** k for x in xs) / n for k in (2, 3, 4))
    g1, g2 = m3 / m2**1.5, m4 / m2**2 - 3
    skew = g1 * math.sqrt(n * (n - 1)) / (n - 2)
    excess_kurtosis = ((n + 1) * g2 + 6) * (n - 1) / ((n - 2) * (n - 3))
    return skew, excess_kurtosis


def test_summary_stats() -> None:
    xs = [1.0, 2.0, 4.0, 8.0, 3.0]
    table = cap.summary_stats(make_artifact(xs)).value
    assert table["count"] == 5
    assert table["mean"] == pytest.approx(3.6)
    assert table["std"] == pytest.approx(_sample_std(xs))
    assert (table["min"], table["max"]) == (1.0, 8.0)
    skew, excess_kurtosis = _adjusted_skew_and_kurtosis(xs)
    assert table["skew"] == pytest.approx(skew)
    assert table["excess_kurtosis"] == pytest.approx(excess_kurtosis)
    assert table["start"] == "2024-01-05"


def test_average_return_of_a_price() -> None:
    """The scenario A chain on a tiny series: prices -> simple returns -> mean."""
    returns = cap.compute_returns(make_artifact([100.0, 102.0, 99.96, 101.958]), "simple")
    # 102/100 - 1 = 0.02, 99.96/102 - 1 = -0.02, 101.958/99.96 - 1 = 0.02 (to rounding)
    assert returns.value.tolist() == pytest.approx([0.02, -0.02, 0.02], abs=1e-4)
    table = cap.summary_stats(returns).value
    assert table["mean"] == pytest.approx(sum(returns.value) / 3)
    assert table["count"] == 3


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


def test_resample_daily_is_a_pass_through() -> None:
    # Mon 2024-01-01 .. Fri 2024-01-12: ten business days, no weekends.
    daily = make_artifact(np.arange(1.0, 11.0), freq=Frequency.D, start="2024-01-01")
    out = cap.resample(daily, freq="D")
    # Unchanged: no weekend rows inserted, no NaN (unlike pandas resample("D")).
    pd.testing.assert_series_equal(out.value, daily.value)
    assert out.meta.frequency == Frequency.D and out.meta.n_obs == 10
    weekly = make_artifact([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="higher frequency"):
        cap.resample(weekly, freq="D")  # weekly -> daily would invent data


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


# --- group comparison -------------------------------------------------------------


def test_compare_groups_averages_each_group_on_common_dates() -> None:
    values = make_artifact([0.01, 0.03, -0.02, 0.04, 0.00], kind=SeriesKind.RETURNS)
    # Flags start one week later, so the first value has no flag and is dropped.
    flags = make_artifact(
        [True, False, True, False], kind=SeriesKind.FLAGS, start="2024-01-12",
        data_type=DataType.FLAGS,
    )  # fmt: skip
    table = cap.compare_groups(values, flags).value
    # Common weeks: 0.03 (flagged), -0.02 (normal), 0.04 (flagged), 0.00 (normal).
    assert (table["n_flagged"], table["n_normal"]) == (2, 2)
    assert table["mean_flagged"] == pytest.approx(0.035)
    assert table["mean_normal"] == pytest.approx(-0.01)
    assert table["diff_means"] == pytest.approx(0.045)
    assert (table["start"], table["end"]) == ("2024-01-12", "2024-02-02")


# Hand-built flags with 3 episodes (runs of True) and values chosen for easy sums.
FLAG_PATTERN = "TTFFTFFFTT"
PATTERN_VALUES = [1.0, 3.0, 2.0, 4.0, 5.0, 6.0, 8.0, 7.0, 9.0, 11.0]


def pattern_inputs(kind: SeriesKind = SeriesKind.RETURNS, **meta: object) -> tuple:
    """(values, flags) artifacts for FLAG_PATTERN on the same ten weeks."""
    values = make_artifact(PATTERN_VALUES, kind=kind, **meta)
    flags = make_artifact(
        [c == "T" for c in FLAG_PATTERN], kind=SeriesKind.FLAGS, data_type=DataType.FLAGS
    )
    return values, flags


def test_compare_groups_std_and_episodes_on_a_hand_built_pattern() -> None:
    table = cap.compare_groups(*pattern_inputs()).value
    # Flagged: 1, 3, 5, 9, 11 (mean 5.8); normal: 2, 4, 6, 8, 7 (mean 5.4).
    assert (table["n_flagged"], table["n_normal"]) == (5, 5)
    assert table["n_episodes_flagged"] == 3  # TT, T, TT: three runs, not five weeks
    assert table["mean_flagged"] == pytest.approx(5.8)
    assert table["mean_normal"] == pytest.approx(5.4)
    # Squared deviations: 23.04+7.84+0.64+10.24+27.04 = 68.8 and
    # 11.56+1.96+0.36+6.76+2.56 = 23.2, each divided by n-1 = 4.
    assert table["std_flagged"] == pytest.approx(math.sqrt(68.8 / 4))
    assert table["std_normal"] == pytest.approx(math.sqrt(23.2 / 4))


def test_compare_groups_refuses_an_empty_group() -> None:
    values = make_artifact([0.01, 0.02], kind=SeriesKind.RETURNS)
    flags = make_artifact([False, False], kind=SeriesKind.FLAGS, data_type=DataType.FLAGS)
    with pytest.raises(ValueError, match="both groups"):
        cap.compare_groups(values, flags)


# --- filter_period --------------------------------------------------------------


def test_explicit_start_and_end_are_inclusive() -> None:
    # Fridays 2024-01-05 .. 2024-03-08 with values 1..10.
    weekly = make_artifact(np.arange(1.0, 11.0), kind=SeriesKind.RETURNS, return_method="log")
    out = cap.filter_period(weekly, start="2024-01-12", end="2024-02-02")
    assert out.value.tolist() == [2.0, 3.0, 4.0, 5.0]  # 01-12, 01-19, 01-26, 02-02
    assert (out.meta.period_start, out.meta.period_end) == ("2024-01-12", "2024-02-02")
    # Same kind, frequency and return method as the input; the interval is in the notes.
    assert (out.meta.kind, out.meta.frequency) == (SeriesKind.RETURNS, Frequency.W)
    assert out.meta.return_method == "log"
    assert out.meta.notes == ["kept 2024-01-12 to 2024-02-02 inclusive (explicit dates)"]


def test_open_ended_periods_resolve_to_the_data_edges() -> None:
    weekly = make_artifact(np.arange(1.0, 11.0))  # 2024-01-05 .. 2024-03-08
    since = cap.filter_period(weekly, start="2024-02-20")
    assert since.value.tolist() == [8.0, 9.0, 10.0]  # 02-23, 03-01, 03-08
    assert (since.meta.period_start, since.meta.period_end) == ("2024-02-20", "2024-03-08")
    until = cap.filter_period(weekly, end="2024-01-15")
    assert until.value.tolist() == [1.0, 2.0]
    assert (until.meta.period_start, until.meta.period_end) == ("2024-01-05", "2024-01-15")


@pytest.mark.parametrize(
    "params",
    [
        {},  # neither start nor end
        {"start": "2024-02-02", "end": "2024-01-12"},  # start after end
    ],
)
def test_filter_period_refuses_an_ambiguous_interval(params: dict) -> None:
    with pytest.raises(ValueError):
        cap.filter_period(make_artifact(np.arange(1.0, 11.0)), **params)


def test_filter_period_metadata_is_not_inherited() -> None:
    weekly = make_artifact(np.arange(1.0, 30.0))
    filtered = cap.filter_period(weekly, start="2024-02-01")
    returns = cap.compute_returns(filtered, method="simple")
    assert returns.meta.period_start is None and returns.meta.period_end is None


# --- find_extremes ----------------------------------------------------------------


def test_find_extremes_max_and_min() -> None:
    out = cap.find_extremes(make_artifact([3.0, 7.0, 1.0, 5.0]))  # Fridays from 2024-01-05
    assert out.value == {
        "max": {"date": "2024-01-12", "value": 7.0, "tie_count": 1},
        "min": {"date": "2024-01-19", "value": 1.0, "tie_count": 1},
    }
    assert out.data_type == DataType.TABLE and out.meta.kind is None
    assert out.meta.source_dataset_ids == ["TEST"]


def test_find_extremes_one_side_on_any_kind() -> None:
    volatility = make_artifact([0.2, 0.5, 0.3], kind=SeriesKind.VOLATILITY)
    assert cap.find_extremes(volatility, which="max").value == {
        "max": {"date": "2024-01-12", "value": 0.5, "tie_count": 1}
    }
    assert list(cap.find_extremes(volatility, which="min").value) == ["min"]


def test_find_extremes_ignores_missing_values() -> None:
    out = cap.find_extremes(make_artifact([np.nan, 2.0, np.nan, 9.0, 4.0]))
    assert out.value["max"] == {"date": "2024-01-26", "value": 9.0, "tie_count": 1}
    assert out.value["min"] == {"date": "2024-01-12", "value": 2.0, "tie_count": 1}
    assert out.meta.n_obs == 3


def test_find_extremes_reports_the_earliest_tie_and_counts_it() -> None:
    out = cap.find_extremes(make_artifact([5.0, 2.0, 5.0, 2.0, 5.0]))
    assert out.value["max"] == {"date": "2024-01-05", "value": 5.0, "tie_count": 3}
    assert out.value["min"] == {"date": "2024-01-12", "value": 2.0, "tie_count": 2}


def test_find_extremes_refuses_an_all_missing_series() -> None:
    with pytest.raises(ValueError):
        cap.find_extremes(make_artifact([np.nan, np.nan]))


# --- forward returns ----------------------------------------------------------------


def test_forward_return_one_period_is_stamped_at_its_start() -> None:
    prices = make_artifact([100.0, 110.0, 99.0, 121.0])
    out = cap.forward_return(prices, method="simple", horizon=1)
    # Value at week t = P_{t+1} / P_t - 1; the last week has no next price and is dropped.
    assert out.value.tolist() == pytest.approx([0.10, -0.10, 121 / 99 - 1])
    assert out.value.index.equals(prices.value.index[:3])
    log = cap.forward_return(prices, method="log", horizon=1)
    assert log.value.tolist() == pytest.approx([math.log(1.1), math.log(0.9), math.log(121 / 99)])
    assert (out.meta.kind, out.meta.horizon, log.meta.return_method) == (
        SeriesKind.FORWARD_RETURNS, 1, "log"
    )  # fmt: skip


def test_forward_return_three_periods_drops_the_last_three() -> None:
    prices = make_artifact([100.0, 110.0, 99.0, 121.0, 132.0])
    out = cap.forward_return(prices, method="simple", horizon=3)
    # 121/100 - 1 at week 0 and 132/110 - 1 at week 1; weeks 2-4 have no price 3 weeks later.
    assert out.value.tolist() == pytest.approx([0.21, 0.20])
    assert out.value.index.equals(prices.value.index[:2])
    assert (out.meta.n_obs, out.meta.horizon, out.meta.return_method) == (2, 3, "simple")
    assert not out.value.isna().any()
    log = cap.forward_return(prices, method="log", horizon=3)
    assert log.value.tolist() == pytest.approx([math.log(1.21), math.log(1.2)])


def test_forward_return_rejects_non_positive_levels() -> None:
    with pytest.raises(ValueError, match="positive"):
        cap.forward_return(make_artifact([1.0, 0.0, 2.0]), method="log")


# --- hac_mean_difference ------------------------------------------------------------


def test_newey_west_lag_rule() -> None:
    # floor(4 * (T/100)^(2/9)): exactly 4 at T=100; 4 * 0.1^(2/9) = 2.40 at T=10;
    # 4 * 10^(2/9) = 6.67 at T=1000.
    assert cap.newey_west_lags(100) == 4
    assert cap.newey_west_lags(10) == 2
    assert cap.newey_west_lags(1000) == 6


def test_hac_coefficients_are_the_group_means() -> None:
    test = cap.hac_mean_difference(*pattern_inputs()).value
    # b = mean(flagged) - mean(normal) = 5.8 - 5.4, and a = mean(normal).
    assert test["diff_means"] == pytest.approx(0.4)
    assert test["mean_normal"] == pytest.approx(5.4)
    assert (test["n_obs"], test["n_flagged"], test["n_normal"]) == (10, 5, 5)
    assert test["n_episodes_flagged"] == 3
    assert test["maxlags_used"] == 2  # floor(4 * 0.1^(2/9)) for T = 10
    assert "horizon" not in test  # plain returns have no horizon


def test_hac_standard_error_matches_a_direct_statsmodels_fit() -> None:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.001, 0.02, 300)
    flagged = np.repeat(rng.random(30) < 0.3, 10)  # clustered flags: 10-week runs
    outcome = make_artifact(returns, kind=SeriesKind.RETURNS)
    flags = make_artifact(flagged, kind=SeriesKind.FLAGS, data_type=DataType.FLAGS)
    for maxlags in (None, 0, 3):
        test = cap.hac_mean_difference(outcome, flags, maxlags=maxlags).value
        lags = cap.newey_west_lags(300) if maxlags is None else maxlags
        design = sm.add_constant(flagged.astype(float))
        direct = sm.OLS(returns, design).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        assert test["maxlags_used"] == lags
        assert test["diff_means"] == pytest.approx(direct.params[1])
        assert test["se_hac"] == pytest.approx(direct.bse[1])
        assert test["t_stat"] == pytest.approx(direct.tvalues[1])
        assert test["p_value"] == pytest.approx(direct.pvalues[1])
        ci_low, ci_high = direct.conf_int(alpha=0.05)[1]
        assert (test["ci_low"], test["ci_high"]) == pytest.approx((ci_low, ci_high))


def test_hac_default_lags_cover_the_forward_return_overlap() -> None:
    def lags(horizon: int, maxlags: int | None = None) -> int:
        outcome, flags = pattern_inputs(SeriesKind.FORWARD_RETURNS, horizon=horizon)
        return cap.hac_mean_difference(outcome, flags, maxlags=maxlags).value["maxlags_used"]

    assert lags(1) == 2  # the rule alone (T = 10)
    assert lags(3) == 2  # h-1 = 2 is already covered by the rule
    assert lags(13) == 12  # raised to h-1
    assert lags(13, maxlags=1) == 1  # a value set in the plan is used as given (G17 warns)
    outcome, flags = pattern_inputs(SeriesKind.FORWARD_RETURNS, horizon=13)
    assert cap.hac_mean_difference(outcome, flags).value["horizon"] == 13


def test_hac_refuses_an_empty_group() -> None:
    outcome = make_artifact([0.01, 0.02, 0.03], kind=SeriesKind.RETURNS)
    flags = make_artifact([True, True, True], kind=SeriesKind.FLAGS, data_type=DataType.FLAGS)
    with pytest.raises(ValueError, match="both groups"):
        cap.hac_mean_difference(outcome, flags)
