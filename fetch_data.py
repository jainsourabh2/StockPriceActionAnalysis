"""
Fetches historical OHLCV candle data from AngelOne SmartAPI.

Usage:
    # Default (RELIANCE, 30 days daily)
    python fetch_data.py

    # Specify a stock
    python fetch_data.py --symbol INFY --token 1594
"""

import argparse
import sys
from datetime import datetime, timedelta
from typing import Optional

from connect import get_session


def fetch_candles(
    smart_api,
    symbol: str = "RELIANCE",
    token: str = "2885",
    exchange: str = "NSE",
    interval: str = "ONE_DAY",
    days: int = 30,
) -> Optional[list]:
    """
    Fetch historical candle data for a stock.

    Args:
        smart_api: Authenticated SmartConnect session
        symbol: Stock symbol (e.g., "RELIANCE")
        token: SmartAPI symbol token (e.g., "2885")
        exchange: Exchange — "NSE" or "BSE"
        interval: Candle interval — "ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE",
                  "THIRTY_MINUTE", "ONE_HOUR", "ONE_DAY"
        days: Number of days of history to fetch

    Returns:
        List of candle data [timestamp, open, high, low, close, volume]
        or None if the request failed.
    """
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)

    params = {
        "exchange": exchange,
        "symboltoken": token,
        "interval": interval,
        "fromdate": from_date.strftime("%Y-%m-%d 09:15"),
        "todate": to_date.strftime("%Y-%m-%d 15:30"),
    }

    print(f"Fetching {days} days of daily data for {symbol}...")

    try:
        response = smart_api.getCandleData(params)
    except Exception as e:
        print(f"API call failed: {e}")
        return None

    if response is None:
        print("Got empty response from SmartAPI. Check your symbol token.")
        return None

    if response.get("status") is False:
        print(f"API error: {response.get('message', 'Unknown error')}")
        return None

    candles = response.get("data")
    if not candles:
        print("No candle data returned. The symbol token might be wrong,")
        print("or the market might be closed for the requested period.")
        return None

    return candles


def print_candles(candles: list, symbol: str) -> None:
    """Print candle data as a clean table."""
    header = f"{'Date':<12} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>12}"
    separator = "─" * len(header)

    print(f"\n{symbol} — Daily OHLCV ({len(candles)} candles)\n")
    print(header)
    print(separator)

    for candle in candles:
        # candle format: [timestamp, open, high, low, close, volume]
        ts = candle[0]
        # Parse the timestamp — SmartAPI returns "2025-01-15T00:00:00+05:30" format
        date_str = ts[:10] if isinstance(ts, str) else str(ts)[:10]
        open_p = candle[1]
        high_p = candle[2]
        low_p = candle[3]
        close_p = candle[4]
        volume = candle[5]

        print(
            f"{date_str:<12} {open_p:>10.2f} {high_p:>10.2f} "
            f"{low_p:>10.2f} {close_p:>10.2f} {volume:>12,}"
        )


def main():
    parser = argparse.ArgumentParser(description="Fetch stock candle data from AngelOne")
    parser.add_argument("--symbol", default="RELIANCE", help="Stock symbol (default: RELIANCE)")
    parser.add_argument("--token", default="2885", help="SmartAPI symbol token (default: 2885)")
    parser.add_argument("--days", type=int, default=30, help="Days of history (default: 30)")
    args = parser.parse_args()

    smart_api = get_session()
    candles = fetch_candles(smart_api, symbol=args.symbol, token=args.token, days=args.days)

    if candles:
        print_candles(candles, args.symbol)
        print(f"\nFetched {len(candles)} candles.")
    else:
        print("Failed to fetch data.")
        sys.exit(1)


if __name__ == "__main__":
    main()
