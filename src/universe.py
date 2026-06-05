"""Insurance stock universe and institutional investor registry.

Defines the set of insurance equities to track and the institutions
whose 13F filings we parse for holdings data.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Insurance stock universe
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Stock:
    ticker: str
    name: str
    cusip: str         # 9-digit CUSIP for 13F matching
    sub_sector: str    # personal, commercial, reinsurance, specialty, multi-line


INSURANCE_UNIVERSE: dict[str, Stock] = {
    "PGR":  Stock("PGR",  "Progressive",              "743315103", "personal"),
    "TRV":  Stock("TRV",  "Travelers Companies",      "89417E109", "multi-line"),
    "ALL":  Stock("ALL",  "Allstate",                  "020002101", "personal"),
    "CB":   Stock("CB",   "Chubb Limited",             "H1467J104", "multi-line"),
    "AIG":  Stock("AIG",  "American International",    "026874784", "multi-line"),
    "HIG":  Stock("HIG",  "Hartford Financial",        "416515104", "commercial"),
    "CNA":  Stock("CNA",  "CNA Financial",             "126117100", "commercial"),
    "MKL":  Stock("MKL",  "Markel Group",              "570535104", "specialty"),
    "RNR":  Stock("RNR",  "RenaissanceRe",             "G7496G103", "reinsurance"),
    "ACGL": Stock("ACGL", "Arch Capital",              "G0450A105", "reinsurance"),
    "AFG":  Stock("AFG",  "American Financial Group",  "025932104", "specialty"),
    "WRB":  Stock("WRB",  "Berkley (W.R.)",            "084423102", "commercial"),
    "RE":   Stock("RE",   "Everest Group",             "G3223R108", "reinsurance"),
    "ERIE": Stock("ERIE", "Erie Indemnity",            "29530P102", "personal"),
    "KNSL": Stock("KNSL", "Kinsale Capital",           "49714P108", "specialty"),
    "CINF": Stock("CINF", "Cincinnati Financial",      "172062101", "multi-line"),
    "THG":  Stock("THG",  "Hanover Insurance",         "410867105", "commercial"),
    "ORI":  Stock("ORI",  "Old Republic International","680223104", "multi-line"),
    "SIGI": Stock("SIGI", "Selective Insurance",       "816300107", "commercial"),
    "RLI":  Stock("RLI",  "RLI Corp",                  "749607107", "specialty"),
    "PLMR": Stock("PLMR", "Palomar Holdings",         "69753M105", "specialty"),
    "RYAN": Stock("RYAN", "Ryan Specialty Holdings",   "78351F107", "specialty"),
}

# Quick lookup sets
ALL_TICKERS = sorted(INSURANCE_UNIVERSE.keys())
ALL_CUSIPS = {s.cusip: s.ticker for s in INSURANCE_UNIVERSE.values()}


def cusip_to_ticker(cusip: str) -> str | None:
    """Look up a ticker from a 9-digit CUSIP. Returns None if not in universe."""
    return ALL_CUSIPS.get(cusip[:9])


def get_stock(ticker: str) -> Stock:
    ticker = ticker.upper()
    if ticker not in INSURANCE_UNIVERSE:
        raise ValueError(f"Unknown ticker: {ticker}")
    return INSURANCE_UNIVERSE[ticker]


def get_stocks_by_sector(sub_sector: str) -> list[Stock]:
    return [s for s in INSURANCE_UNIVERSE.values() if s.sub_sector == sub_sector]


# ---------------------------------------------------------------------------
# Institutional investor registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Institution:
    name: str
    cik: str           # SEC CIK for EDGAR lookups
    style: str         # "passive" or "active"
    aum_approx_b: float  # Approximate AUM in billions (for weighting)


INSTITUTION_REGISTRY: dict[str, Institution] = {
    "blackrock": Institution(
        name="BlackRock",
        cik="0001364742",
        style="passive",
        aum_approx_b=10000,
    ),
    "vanguard": Institution(
        name="Vanguard Group",
        cik="0000102909",
        style="passive",
        aum_approx_b=8500,
    ),
    "state_street": Institution(
        name="State Street Corporation",
        cik="0000093751",
        style="passive",
        aum_approx_b=4100,
    ),
    "berkshire": Institution(
        name="Berkshire Hathaway",
        cik="0001067983",
        style="active",
        aum_approx_b=350,
    ),
    "capital_group": Institution(
        name="Capital Group",
        cik="0000080255",
        style="active",
        aum_approx_b=2600,
    ),
    "wellington": Institution(
        name="Wellington Management",
        cik="0000102426",
        style="active",
        aum_approx_b=1200,
    ),
    "t_rowe": Institution(
        name="T. Rowe Price",
        cik="0001015308",
        style="active",
        aum_approx_b=1500,
    ),
    "fidelity": Institution(
        name="FMR LLC (Fidelity)",
        cik="0000315066",
        style="active",
        aum_approx_b=4500,
    ),
    "dimensional": Institution(
        name="Dimensional Fund Advisors",
        cik="0000354204",
        style="passive",
        aum_approx_b=700,
    ),
}

PASSIVE_INSTITUTIONS = [k for k, v in INSTITUTION_REGISTRY.items() if v.style == "passive"]
ACTIVE_INSTITUTIONS = [k for k, v in INSTITUTION_REGISTRY.items() if v.style == "active"]


def get_institution(key: str) -> Institution:
    key = key.lower()
    if key not in INSTITUTION_REGISTRY:
        raise ValueError(f"Unknown institution: {key}")
    return INSTITUTION_REGISTRY[key]
