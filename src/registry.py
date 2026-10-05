"""
registry.py — Single registry of the funds in the corpus

Built from metadata.json (one entry per PDF): documents are grouped by ISIN
into one Fund each. Every module that needs ISINs, names, issuers or Yahoo
tickers (agent tools, ask.py, live_data.py) reads them from here, so adding
an ETF only means adding its documents to metadata.json.

    from registry import FUNDS, get_fund, resolve_isins

    FUNDS["IE00B4L5Y983"].name           # "iShares Core MSCI World UCITS ETF"
    get_fund("ie00b4l5y983").yahoo_ticker  # "EUNL.DE"
    resolve_isins(issuer="ubs")          # ["IE00BD4TXV59"]

Usage (CLI — print the registry):
    python src/registry.py
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import METADATA_PATH


@dataclass
class Fund:
    isin: str
    name: str
    issuer: str
    category: str
    currency: str = ""
    yahoo_ticker: Optional[str] = None
    doc_types: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)

    def describe(self) -> str:
        docs  = ", ".join(self.doc_types)
        years = ", ".join(str(y) for y in self.years)
        return (f"{self.name} | ISIN: {self.isin} | issuer: {self.issuer} | "
                f"category: {self.category} | documents: {docs} ({years})")


# Fund-level fields that must agree across all documents of the same ISIN
_FUND_FIELDS = ("name", "issuer", "category", "currency", "yahoo_ticker")


def load_registry(path: Path = METADATA_PATH) -> dict[str, Fund]:
    """ISIN → Fund, in metadata.json order. Raises ValueError on inconsistent entries."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    funds: dict[str, Fund] = {}

    for filename, entry in raw.items():
        isin = entry["isin"].upper()
        doc_type = entry.get("doc_type") or entry.get("type", "factsheet")
        values = {
            "name":         entry.get("name") or isin,
            "issuer":       entry["issuer"].lower(),
            "category":     entry.get("category", ""),
            "currency":     entry.get("currency", ""),
            "yahoo_ticker": entry.get("yahoo_ticker") or None,
        }

        fund = funds.get(isin)
        if fund is None:
            fund = funds[isin] = Fund(isin=isin, **values)
        else:
            for key in _FUND_FIELDS:
                if values[key] != getattr(fund, key):
                    raise ValueError(
                        f"metadata.json: '{filename}' has {key}={values[key]!r} but another "
                        f"document of {isin} has {getattr(fund, key)!r}"
                    )

        if doc_type not in fund.doc_types:
            fund.doc_types.append(doc_type)
        year = int(entry["year"])
        if year not in fund.years:
            fund.years.append(year)

    for fund in funds.values():
        fund.doc_types.sort()
        fund.years.sort()
    return funds


FUNDS: dict[str, Fund] = load_registry()


# ── Lookups ────────────────────────────────────────────────────────────────────

def get_fund(isin: str) -> Optional[Fund]:
    return FUNDS.get(isin.strip().upper())


def all_isins() -> list[str]:
    return list(FUNDS)


def all_issuers() -> list[str]:
    return sorted({f.issuer for f in FUNDS.values()})


def all_doc_types() -> list[str]:
    return sorted({d for f in FUNDS.values() for d in f.doc_types})


def all_years() -> list[int]:
    return sorted({y for f in FUNDS.values() for y in f.years})


def resolve_isins(isins: Optional[list[str]] = None,
                  issuer: Optional[str] = None) -> list[str]:
    """
    The ISINs selected by an explicit list and/or an issuer (intersection when
    both are given); every fund when neither is. Raises ValueError on unknown values.
    """
    selected = all_isins()
    if isins:
        wanted  = [i.strip().upper() for i in isins]
        unknown = [i for i in wanted if i not in FUNDS]
        if unknown:
            raise ValueError(f"Unknown ISIN(s) {unknown}. Known: {', '.join(FUNDS)}")
        selected = list(dict.fromkeys(wanted))
    if issuer:
        issuer = issuer.lower()
        if issuer not in all_issuers():
            raise ValueError(f"Unknown issuer {issuer!r}. Known: {', '.join(all_issuers())}")
        selected = [i for i in selected if FUNDS[i].issuer == issuer]
    return selected


if __name__ == "__main__":
    for f in FUNDS.values():
        print(f"{f.describe()} | yahoo: {f.yahoo_ticker or '-'}")
