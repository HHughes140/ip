"""Tests for Axioma factor snapshot building (pure functions, no Snowflake)."""

import pandas as pd

from src.data.snowflake_factors import build_quarterly_snapshots, AXIOMA_FACTORS


def _make_daily() -> pd.DataFrame:
    """Two tickers, daily exposures spanning two quarters of 2024."""
    dates = pd.bdate_range("2024-01-02", "2024-06-28")
    records = []
    for ticker in ["PGR", "TRV"]:
        for i, date in enumerate(dates):
            record = {"ticker": ticker, "date": date}
            for j, factor in enumerate(AXIOMA_FACTORS):
                record[factor] = (i + j) * 0.01 * (1 if ticker == "PGR" else -1)
            records.append(record)
    return pd.DataFrame(records)


class TestBuildQuarterlySnapshots:
    def test_empty_input(self):
        assert build_quarterly_snapshots(pd.DataFrame()).empty

    def test_one_row_per_ticker_quarter(self):
        snapshots = build_quarterly_snapshots(_make_daily())
        assert len(snapshots) == 4  # 2 tickers × 2 quarters
        assert not snapshots.duplicated(subset=["ticker", "quarter"]).any()

    def test_quarter_format(self):
        snapshots = build_quarterly_snapshots(_make_daily())
        assert set(snapshots["quarter"]) == {"2024-Q1", "2024-Q2"}

    def test_all_factors_present(self):
        snapshots = build_quarterly_snapshots(_make_daily())
        for factor in AXIOMA_FACTORS:
            assert factor in snapshots.columns

    def test_uses_quarter_end_observation(self):
        daily = _make_daily()
        snapshots = build_quarterly_snapshots(daily)

        # PGR Q1: snapshot should match the last Q1 date's value
        q1_dates = daily[(daily["ticker"] == "PGR") & (daily["date"] < "2024-04-01")]
        expected = q1_dates.sort_values("date")["value"].iloc[-1]
        actual = snapshots[
            (snapshots["ticker"] == "PGR") & (snapshots["quarter"] == "2024-Q1")
        ]["value"].iloc[0]
        assert actual == expected
