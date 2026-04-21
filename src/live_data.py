"""
live_data.py — Live market data tool for FundScope
Fetches current price, returns, volume, and AUM via yfinance.

This module is designed to be called as a tool by the agent loop in agent.py.
It is not meant to be called directly from ask.py or generate.py.

Install:
    pip install yfinance

Usage (standalone test):
    python src/live_data.py --isin IE00B4L5Y983
    python src/live_data.py --ticker EUNL.DE
"""

import argparse
from datetime import datetime, timedelta
from typing import Optional


try:
    import yfinance as yf
except ImportError:
    raise ImportError(
        "pip install yfinance\n"
        "yfinance is required for live market data."
    )


# ── Ticker map ─────────────────────────────────────────────────────────────────
# Maps ISINs to Yahoo Finance tickers.
# Yahoo Finance does not support ISIN lookups directly — this map bridges them.
# Extend this as you add more ETFs to the corpus.

ISIN_TO_TICKER: dict[str, str] = {
    "IE00B4L5Y983": "EUNL.DE",   # iShares Core MSCI World UCITS ETF (EUR, Xetra)
    "IE00BD4TXV59": "0P0001FMRI.L",  # UBS Core MSCI World (GBP, London — best available)
}

# Human-readable names for display
ISIN_TO_NAME: dict[str, str] = {
    "IE00B4L5Y983": "iShares Core MSCI World",
    "IE00BD4TXV59": "UBS Core MSCI World",
}


# ── Core fetch ─────────────────────────────────────────────────────────────────

def get_etf_live_data(isin_or_ticker: str, period_months: int = 3) -> dict:
    """
    Fetch live price and performance data for an ETF.

    Args:
        isin_or_ticker: An ISIN (looked up via ISIN_TO_TICKER) or a raw ticker string.
        period_months:  How many months of history to fetch for return calculations.

    Returns:
        A flat dict with all fields ready for prompt injection or display.
        On failure, returns {"error": "...", "isin_or_ticker": "..."}.
    """
    # Resolve ticker
    if isin_or_ticker.upper() in ISIN_TO_TICKER:
        isin   = isin_or_ticker.upper()
        ticker = ISIN_TO_TICKER[isin]
        name   = ISIN_TO_NAME.get(isin, isin)
    else:
        isin   = None
        ticker = isin_or_ticker
        name   = ticker

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
    current_price = hist["Close"].iloc[-1]
    start_price   = hist["Close"].iloc[0]
    period_return = (current_price - start_price) / start_price * 100

    # 1-month return (subset of history)
    cutoff_1m = hist.index[-1] - timedelta(days=30)
    hist_1m   = hist[hist.index >= cutoff_1m]
    one_month_return = (
        (current_price - hist_1m["Close"].iloc[0]) / hist_1m["Close"].iloc[0] * 100
        if not hist_1m.empty else None
    )

    # 52-week high/low (from info if available, else compute from history)
    week52_high = info.get("fiftyTwoWeekHigh") or float(hist["High"].max())
    week52_low  = info.get("fiftyTwoWeekLow")  or float(hist["Low"].min())

    # ── Volume ────────────────────────────────────────────────────────────────
    avg_volume = int(hist["Volume"].mean()) if hist["Volume"].sum() > 0 else None

    # ── Build result ──────────────────────────────────────────────────────────
    return {
        # Identification
        "name":             name,
        "isin":             isin,
        "ticker":           ticker,
        "as_of":            hist.index[-1].strftime("%Y-%m-%d"),

        # Price
        "current_price":    round(float(current_price), 4),
        "currency":         info.get("currency", "N/A"),
        "week52_high":      round(float(week52_high), 4) if week52_high else None,
        "week52_low":       round(float(week52_low), 4)  if week52_low  else None,

        # Returns
        "return_1m_pct":    round(float(one_month_return), 2) if one_month_return is not None else None,
        "return_3m_pct":    round(float(period_return), 2),
        "return_ytd":       info.get("ytdReturn"),           # float or None
        "return_3y_avg":    info.get("threeYearAverageReturn"),
        "return_5y_avg":    info.get("fiveYearAverageReturn"),

        # Size & liquidity
        "total_assets":     info.get("totalAssets"),         # AUM in fund currency
        "avg_volume":       avg_volume,

        # Fund metadata from Yahoo
        "fund_family":      info.get("fundFamily"),
        "category":         info.get("category"),
        "exchange":         info.get("exchange"),
    }


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
    p.add_argument("--months", type=int, default=3,
                   help="Months of history for return calculations (default: 3)")
    return p


def main():
    args   = _build_parser().parse_args()
    target = args.isin or args.ticker
    data   = get_etf_live_data(target, period_months=args.months)
    print(format_for_prompt(data))
    print()
    # Also print raw dict for debugging
    import json
    print("Raw data:")
    print(json.dumps(data, indent=2, default=str))


if __name__ == "__main__":
    main()
