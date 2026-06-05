"""Tests for 13F parsing and delta computation."""

import pandas as pd
import pytest

from src.data.edgar_13f import compute_deltas, _parse_info_table


class TestComputeDeltas:
    def test_basic_delta(self):
        holdings = pd.DataFrame({
            "institution": ["fidelity", "fidelity", "fidelity", "fidelity"],
            "ticker": ["PGR", "PGR", "TRV", "TRV"],
            "quarter": ["2024-Q1", "2024-Q2", "2024-Q1", "2024-Q2"],
            "shares": [100000, 120000, 50000, 45000],
            "value": [10000, 13000, 8000, 7500],
            "style": ["active", "active", "active", "active"],
        })
        result = compute_deltas(holdings)
        # PGR Q2 delta should be +20000
        pgr_q2 = result[(result["ticker"] == "PGR") & (result["quarter"] == "2024-Q2")]
        assert pgr_q2["delta_shares"].iloc[0] == 20000
        # TRV Q2 delta should be -5000
        trv_q2 = result[(result["ticker"] == "TRV") & (result["quarter"] == "2024-Q2")]
        assert trv_q2["delta_shares"].iloc[0] == -5000

    def test_first_quarter_is_nan(self):
        holdings = pd.DataFrame({
            "institution": ["fidelity", "fidelity"],
            "ticker": ["PGR", "PGR"],
            "quarter": ["2024-Q1", "2024-Q2"],
            "shares": [100000, 120000],
            "value": [10000, 13000],
            "style": ["active", "active"],
        })
        result = compute_deltas(holdings)
        q1 = result[result["quarter"] == "2024-Q1"]
        assert pd.isna(q1["delta_shares"].iloc[0])

    def test_portfolio_weight(self):
        holdings = pd.DataFrame({
            "institution": ["fidelity", "fidelity"],
            "ticker": ["PGR", "TRV"],
            "quarter": ["2024-Q1", "2024-Q1"],
            "shares": [100000, 50000],
            "value": [10000, 5000],
            "style": ["active", "active"],
        })
        result = compute_deltas(holdings)
        # PGR should be 66.7% of portfolio
        pgr = result[result["ticker"] == "PGR"]
        assert pgr["portfolio_weight"].iloc[0] == pytest.approx(66.67, rel=0.01)

    def test_empty_input(self):
        result = compute_deltas(pd.DataFrame())
        assert result.empty


class TestParseInfoTable:
    def test_namespaced_xml(self):
        xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
            <infoTable>
                <nameOfIssuer>PROGRESSIVE CORP</nameOfIssuer>
                <cusip>743315103</cusip>
                <value>150000</value>
                <shrsOrPrnAmt>
                    <sshPrnamt>1000000</sshPrnamt>
                    <sshPrnamtType>SH</sshPrnamtType>
                </shrsOrPrnAmt>
                <investmentDiscretion>SOLE</investmentDiscretion>
            </infoTable>
        </informationTable>"""

        holdings = _parse_info_table(xml)
        assert len(holdings) == 1
        assert holdings[0]["cusip"] == "743315103"
        assert holdings[0]["shares"] == 1000000
        assert holdings[0]["value"] == 150000

    def test_empty_xml(self):
        xml = b"""<?xml version="1.0"?><root></root>"""
        holdings = _parse_info_table(xml)
        assert len(holdings) == 0
