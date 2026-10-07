"""
live_data.py — Live market data tool for FundScope
Fetches current price, returns, volume, and AUM via yfinance.

This module is designed to be called as a tool by the agent loop in agent.py.
It is not meant to be called directly from ask.py or generate.py.

Install:
    pip install yfinance

Usage (standalone test):
    python src/live_data.py --isin IE00B4L5Y983
    python src/live_data.py --ticker EUNL.DE [--no_cache]

Successful answers are cached on disk for config.LIVE_DATA_TTL_S seconds
(default 1 h) in config.LIVE_DATA_CACHE_DIR; errors are never cached.
"""

import argparse
import json
import re
import time
from datetime import timedelta
from typing import Optional


try:
    import yfinance as yf
except ImportError:
    raise ImportError(
        "pip install yfinance\n"
        "yfinance is required for live market data."
    )

from config import LIVE_DATA_CACHE_DIR, LIVE_DATA_TTL_S
from registry import get_fund


# ── Disk cache ─────────────────────────────────────────────────────────────────

def _cache_path(ticker: str, period_months: int):
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", ticker)
    return LIVE_DATA_CACHE_DIR / f"{safe}_{period_months}m.json"


def _cache_read(ticker: str, period_months: int) -> Optional[dict]:
    if LIVE_DATA_TTL_S <= 0:
        return None
    try:
        entry = json.loads(_cache_path(ticker, period_months).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if time.time() - entry.get("fetched_at", 0) > LIVE_DATA_TTL_S:
        return None
    return entry.get("data")


def _cache_write(ticker: str, period_months: int, data: dict) -> None:
    if LIVE_DATA_TTL_S <= 0:
        return
    path = _cache_path(ticker, period_months)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"fetched_at": time.time(), "data": data}, default=str),
                       encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass   # a cache failure must never break the tool


# ── Return helpers ─────────────────────────────────────────────────────────────

def _fraction_to_pct(val) -> Optional[float]:
    """Yahoo's threeYearAverageReturn / fiveYearAverageReturn are fractions (0.12 = 12%)."""
    return round(float(val) * 100, 2) if val is not None else None


def _return_since(hist, start, current_price) -> Optional[float]:
    """% return from the last close before `start` (or the first close after it) to now."""
    before = hist[hist.index < start]
    if not before.empty:
        base = before["Close"].iloc[-1]
    else:
        after = hist[hist.index >= start]
        if after.empty:
            return None
        base = after["Close"].iloc[0]
    return round(float((current_price - base) / base * 100), 2)


# ── Core fetch ─────────────────────────────────────────────────────────────────

def get_etf_live_data(isin_or_ticker: str, period_months: int = 12,
                      use_cache: bool = True) -> dict:
    """
    Fetch live price and performance data for an ETF.

    Args:
        isin_or_ticker: An ISIN (its Yahoo ticker comes from the fund registry,
                        i.e. `yahoo_ticker` in metadata.json) or a raw ticker string.
        period_months:  Months of price history to fetch (at least 12, so the
                        52-week range and the YTD return can be computed from it).
        use_cache:      Read/write the on-disk cache (TTL config.LIVE_DATA_TTL_S).

    Returns:
        A flat dict with all fields ready for prompt injection or display; every
        return is in percent (12.3 = +12.3%). On failure, returns
        {"error": "...", "isin_or_ticker": "..."} (never cached).
    """
    period_months = max(period_months, 12)
    # Resolve ticker
    fund = get_fund(isin_or_ticker)
    if fund is not None:
        if not fund.yahoo_ticker:
            return {"error": "No Yahoo ticker registered (yahoo_ticker in metadata.json).",
                    "isin_or_ticker": isin_or_ticker}
        isin   = fund.isin
        ticker = fund.yahoo_ticker
        name   = fund.name
    else:
        isin   = None
        ticker = isin_or_ticker
        name   = ticker

    if use_cache:
        cached = _cache_read(ticker, period_months)
        if cached is not None:
            return cached

    try:
        etf  = yf.Ticker(ticker)
        hist = etf.history(period=f"{period_months}mo")
        info = etf.info or {}
    except Exception as e:
        return {"error": str(e), "isin_or_ticker": isin_or_ticker}

    if hist.empty:
        return {
            "error": "No price data returned — ticker may be delisted or incorrect.",
            "isin_or_ticker": isin_or_ticker,
            "ticker": ticker,
        }

    # ── Prices ────────────────────────────────────────────────────────────────
    last_date     = hist.index[-1]
    current_price = hist["Close"].iloc[-1]

    one_month_return   = _return_since(hist, last_date - timedelta(days=30), current_price)
    three_month_return = _return_since(hist, last_date - timedelta(days=91), current_price)

    # Task 6.2 — units checked on yfinance 1.7.0 (URTH, SPY): `ytdReturn` is
    # already in percent (11.74), the 3y/5y averages are fractions (0.2245).
    # European listings (EUNL.DE, IWDA.AS, SWDA.L) return None for all three,
    # so YTD falls back to the price history.
    ytd_return = info.get("ytdReturn")
    if ytd_return is not None:
        ytd_return = round(float(ytd_return), 2)
    else:
        year_start = last_date.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        ytd_return = _return_since(hist, year_start, current_price)

    # 52-week high/low (from info if available, else from the last year of history)
    hist_52w    = hist[hist.index >= last_date - timedelta(days=365)]
    week52_high = info.get("fiftyTwoWeekHigh") or float(hist_52w["High"].max())
    week52_low  = info.get("fiftyTwoWeekLow")  or float(hist_52w["Low"].min())

    # ── Volume (last 3 months) ────────────────────────────────────────────────
    hist_3m    = hist[hist.index >= last_date - timedelta(days=91)]
    avg_volume = int(hist_3m["Volume"].mean()) if hist_3m["Volume"].sum() > 0 else None

    # ── Build result ──────────────────────────────────────────────────────────
    data = {
        # Identification
        "name":             name,
        "isin":             isin,
        "ticker":           ticker,
        "as_of":            last_date.strftime("%Y-%m-%d"),

        # Price
        "current_price":    round(float(current_price), 4),
        "currency":         info.get("currency", "N/A"),
        "week52_high":      round(float(week52_high), 4) if week52_high else None,
        "week52_low":       round(float(week52_low), 4)  if week52_low  else None,

        # Returns, all in percent
        "return_1m_pct":    one_month_return,
        "return_3m_pct":    three_month_return,
        "return_ytd":       ytd_return,
        "return_3y_avg":    _fraction_to_pct(info.get("threeYearAverageReturn")),
        "return_5y_avg":    _fraction_to_pct(info.get("fiveYearAverageReturn")),

        # Size & liquidity
        "total_assets":     info.get("totalAssets"),         # AUM in fund currency
        "avg_volume":       avg_volume,

        # Fund metadata from Yahoo
        "fund_family":      info.get("fundFamily"),
        "category":         info.get("category"),
        "exchange":         info.get("exchange"),
    }
    if use_cache:
        _cache_write(ticker, period_months, data)
    return data


# ── Prompt formatter ───────────────────────────────────────────────────────────

def format_for_prompt(data: dict) -> str:
    """
    Render a live data dict as a clean text block for LLM prompt injection.
    Called by agent.py after executing the get_live_data tool.
    """
    if "error" in data:
        return (
            f"[Live data unavailable for {data.get('ticker', data.get('isin_or_ticker', '?'))}: "
            f"{data['error']}]"
        )

    def fmt_pct(val) -> str:
        return f"{val:+.2f}%" if val is not None else "N/A"

    def fmt_aum(val) -> str:
        if val is None:
            return "N/A"
        if val >= 1_000_000_000:
            return f"{val / 1_000_000_000:.2f}B"
        if val >= 1_000_000:
            return f"{val / 1_000_000:.1f}M"
        return str(val)

    lines = [
        f"=== Live Market Data: {data['name']} ({data['ticker']}) — as of {data['as_of']} ===",
        f"Price:       {data['current_price']} {data['currency']}",
        f"52w High:    {data['week52_high']} {data['currency']}" if data.get("week52_high") else "",
        f"52w Low:     {data['week52_low']} {data['currency']}"  if data.get("week52_low")  else "",
        f"1M Return:   {fmt_pct(data.get('return_1m_pct'))}",
        f"3M Return:   {fmt_pct(data.get('return_3m_pct'))}",
        f"YTD Return:  {fmt_pct(data.get('return_ytd'))}",
        f"3Y Avg:      {fmt_pct(data.get('return_3y_avg'))}",
        f"5Y Avg:      {fmt_pct(data.get('return_5y_avg'))}",
        f"AUM:         {fmt_aum(data.get('total_assets'))}",
        f"Avg Volume:  {data['avg_volume']:,}" if data.get("avg_volume") else "",
    ]
    return "\n".join(l for l in lines if l)


# ── CLI (for quick manual tests) ──────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fetch live ETF data from Yahoo Finance")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--isin",   help="ISIN code, e.g. IE00B4L5Y983")
    group.add_argument("--ticker", help="Yahoo Finance ticker, e.g. EUNL.DE")
    p.add_argument("--months", type=int, default=12,
                   help="Months of price history to fetch (default and minimum: 12)")
    p.add_argument("--no_cache", action="store_true",
                   help="Bypass the on-disk cache (neither read nor written)")
    return p


def main():
    args   = _build_parser().parse_args()
    target = args.isin or args.ticker
    data   = get_etf_live_data(target, period_months=args.months, use_cache=not args.no_cache)
    print(format_for_prompt(data))
    print()
    # Also print raw dict for debugging
    import json
    print("Raw data:")
    print(json.dumps(data, indent=2, default=str))


if __name__ == "__main__":
    main()
