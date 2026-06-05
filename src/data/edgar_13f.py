"""SEC EDGAR 13F-HR filing parser.

Extracts institutional holdings from 13F-HR filings, filters to the
insurance stock universe, and computes quarter-over-quarter position changes.

13F filings use XML infoTable format. Each row contains:
    nameOfIssuer, cusip, value (in thousands), shares/amount, investmentDiscretion

SEC API rate limit: 10 req/s with descriptive User-Agent.
"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from src.data import cache
from src.universe import (
    INSTITUTION_REGISTRY,
    ALL_CUSIPS,
    cusip_to_ticker,
    Institution,
)

logger = logging.getLogger(__name__)

NAMESPACE = "13f"

USER_AGENT = "InstitutionalPressure/0.1 (research tool)"
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

# XML namespaces used in 13F info tables
NS_13F = "http://www.sec.gov/edgar/document/thirteenf/informationtable"

_last_request_time = 0.0


def _sec_get(url: str) -> requests.Response:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < 0.12:
        time.sleep(0.12 - elapsed)
    _last_request_time = time.time()

    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# Filing discovery
# ---------------------------------------------------------------------------

def _get_13f_filings(cik: str, start_year: int = 2015) -> list[dict]:
    """Get list of 13F-HR filing accession numbers from EDGAR submissions API."""
    cik_padded = cik.lstrip("0").zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

    try:
        resp = _sec_get(url)
        data = resp.json()
    except Exception as e:
        logger.error("Failed to get submissions for CIK %s: %s", cik, e)
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    filings = []
    for i, form in enumerate(forms):
        if form in ("13F-HR", "13F-HR/A"):
            filing_year = int(dates[i][:4])
            if filing_year >= start_year:
                filings.append({
                    "form": form,
                    "filing_date": dates[i],
                    "accession": accessions[i],
                    "primary_doc": primary_docs[i],
                    "cik": cik_padded,
                })

    logger.info("Found %d 13F filings for CIK %s", len(filings), cik)
    return filings


def _get_info_table_url(filing: dict) -> str | None:
    """Find the infotable XML document URL within a 13F filing."""
    cik = filing["cik"]
    accession_clean = filing["accession"].replace("-", "")
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik}/"
        f"{accession_clean}/{filing['accession']}-index.htm"
    )

    try:
        resp = _sec_get(index_url)
        text = resp.text
    except Exception:
        # Try the JSON index instead
        json_url = (
            f"https://data.sec.gov/submissions/"
            f"{filing['accession']}.json"
        )
        try:
            resp = _sec_get(json_url)
        except Exception as e:
            logger.warning("Could not get filing index: %s", e)
            return None

    # Look for the info table XML file in the filing index
    # Common names: infotable.xml, primary_doc.xml, etc.
    pattern = re.compile(
        r'href="([^"]*(?:infotable|information_table|INFOTABLE)[^"]*\.xml)"',
        re.IGNORECASE,
    )
    match = pattern.search(resp.text)
    if match:
        rel_path = match.group(1)
        if rel_path.startswith("http"):
            return rel_path
        return (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/"
            f"{accession_clean}/{rel_path}"
        )

    # Fallback: try common naming patterns
    for name in ["infotable.xml", "INFOTABLE.XML", "primary_doc.xml"]:
        test_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/"
            f"{accession_clean}/{name}"
        )
        try:
            test_resp = requests.head(test_url, headers=HEADERS, timeout=10)
            if test_resp.status_code == 200:
                return test_url
        except Exception:
            continue

    logger.warning("Could not find info table XML for %s", filing["accession"])
    return None


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def _parse_info_table(xml_content: bytes) -> list[dict]:
    """Parse a 13F info table XML into a list of holdings records."""
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        logger.warning("XML parse error: %s", e)
        return []

    holdings = []

    # Handle both namespaced and non-namespaced XML
    # Try namespaced first
    info_tables = root.findall(f".//{{{NS_13F}}}infoTable")
    if not info_tables:
        # Try without namespace
        info_tables = root.findall(".//infoTable")
    if not info_tables:
        # Try finding any element with relevant children
        info_tables = root.findall(".//*[cusip]") or root.findall(f".//*[{{{NS_13F}}}cusip]")

    for entry in info_tables:
        def _find(tag: str) -> str:
            # Try with namespace first, then without
            el = entry.find(f"{{{NS_13F}}}{tag}")
            if el is None:
                el = entry.find(tag)
            return el.text.strip() if el is not None and el.text else ""

        def _find_nested(parent_tag: str, child_tag: str) -> str:
            parent = entry.find(f"{{{NS_13F}}}{parent_tag}")
            if parent is None:
                parent = entry.find(parent_tag)
            if parent is None:
                return ""
            child = parent.find(f"{{{NS_13F}}}{child_tag}")
            if child is None:
                child = parent.find(child_tag)
            return child.text.strip() if child is not None and child.text else ""

        cusip = _find("cusip")
        if not cusip:
            continue

        # Normalize CUSIP to 9 chars
        cusip = cusip.replace(" ", "").upper()[:9]

        shares_str = _find_nested("shrsOrPrnAmt", "sshPrnamt")
        value_str = _find("value")

        try:
            shares = int(shares_str.replace(",", "")) if shares_str else 0
        except ValueError:
            shares = 0

        try:
            value_thousands = int(value_str.replace(",", "")) if value_str else 0
        except ValueError:
            value_thousands = 0

        holdings.append({
            "cusip": cusip,
            "name_of_issuer": _find("nameOfIssuer"),
            "shares": shares,
            "value_thousands": value_thousands,
            "investment_discretion": _find("investmentDiscretion"),
            "share_type": _find_nested("shrsOrPrnAmt", "sshPrnamtType"),
        })

    return holdings


# ---------------------------------------------------------------------------
# Holdings extraction
# ---------------------------------------------------------------------------

def extract_holdings(
    institution_key: str,
    data_dir: str,
    start_year: int = 2015,
    force: bool = False,
) -> pd.DataFrame:
    """Extract all insurance stock holdings from an institution's 13F filings.

    Returns DataFrame with columns:
        institution, ticker, filing_date, quarter, shares, value_thousands,
        investment_discretion
    """
    cache_name = f"holdings_{institution_key}"
    if not force and not cache.is_stale(data_dir, NAMESPACE, cache_name, max_age_hours=720):
        cached = cache.load(data_dir, NAMESPACE, cache_name)
        if cached is not None:
            return cached

    inst = INSTITUTION_REGISTRY[institution_key]
    filings = _get_13f_filings(inst.cik, start_year)

    all_records = []

    for filing in filings:
        info_url = _get_info_table_url(filing)
        if not info_url:
            continue

        try:
            resp = _sec_get(info_url)
            holdings = _parse_info_table(resp.content)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", info_url, e)
            continue

        # Filter to insurance universe by CUSIP
        for h in holdings:
            ticker = cusip_to_ticker(h["cusip"])
            if ticker is None:
                continue

            # Determine quarter from filing date
            # 13F reports positions as of quarter-end, filed ~45 days later
            filing_date = filing["filing_date"]
            filing_month = int(filing_date[5:7])
            filing_year = int(filing_date[:4])

            # Filing in Feb → Q4 prior year, May → Q1, Aug → Q2, Nov → Q3
            if filing_month <= 2:
                report_quarter = f"{filing_year - 1}-Q4"
            elif filing_month <= 5:
                report_quarter = f"{filing_year}-Q1"
            elif filing_month <= 8:
                report_quarter = f"{filing_year}-Q2"
            elif filing_month <= 11:
                report_quarter = f"{filing_year}-Q3"
            else:
                report_quarter = f"{filing_year}-Q4"

            all_records.append({
                "institution": institution_key,
                "institution_name": inst.name,
                "style": inst.style,
                "ticker": ticker,
                "quarter": report_quarter,
                "filing_date": filing_date,
                "shares": h["shares"],
                "value_thousands": h["value_thousands"],
                "investment_discretion": h["investment_discretion"],
            })

    if not all_records:
        logger.warning("No insurance holdings found for %s", institution_key)
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df = df.sort_values(["ticker", "quarter"]).reset_index(drop=True)

    cache.save(df, data_dir, NAMESPACE, cache_name)
    logger.info("Extracted %d holdings records for %s", len(df), institution_key)
    return df


def compute_deltas(holdings: pd.DataFrame) -> pd.DataFrame:
    """Compute quarter-over-quarter position changes.

    Adds columns: delta_shares, delta_pct, portfolio_weight
    """
    if holdings.empty:
        return holdings

    df = holdings.sort_values(["institution", "ticker", "quarter"]).copy()

    # Compute delta shares per institution × ticker
    df["delta_shares"] = df.groupby(["institution", "ticker"])["shares"].diff()

    # Compute percentage change
    prev_shares = df.groupby(["institution", "ticker"])["shares"].shift(1)
    df["delta_pct"] = (df["shares"] - prev_shares) / prev_shares.replace(0, float("nan")) * 100

    # Compute portfolio weight (value of position / total portfolio value per quarter)
    quarter_totals = df.groupby(["institution", "quarter"])["value_thousands"].transform("sum")
    df["portfolio_weight"] = df["value_thousands"] / quarter_totals.replace(0, float("nan")) * 100

    return df


def refresh_all(
    data_dir: str,
    start_year: int = 2015,
    force: bool = False,
) -> pd.DataFrame:
    """Extract holdings from all institutions and compute deltas.

    Returns a combined DataFrame with all holdings and position changes.
    """
    all_holdings = []

    for key in INSTITUTION_REGISTRY:
        try:
            df = extract_holdings(key, data_dir, start_year, force)
            if not df.empty:
                all_holdings.append(df)
        except Exception as e:
            logger.error("Failed to process %s: %s", key, e)

    if not all_holdings:
        logger.warning("No holdings extracted from any institution")
        return pd.DataFrame()

    combined = pd.concat(all_holdings, ignore_index=True)
    combined = compute_deltas(combined)

    cache.save(combined, data_dir, NAMESPACE, "all_holdings")
    logger.info("Total holdings: %d records across %d institutions",
                len(combined), combined["institution"].nunique())

    return combined


def load_holdings(data_dir: str) -> pd.DataFrame | None:
    return cache.load(data_dir, NAMESPACE, "all_holdings")
