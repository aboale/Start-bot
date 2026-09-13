"""
Lightweight data providers for the Android app.
Uses only `requests` (no pandas / yfinance / ccxt) to keep the APK small
and avoid heavy native dependencies that are painful to cross-compile
for Android with buildozer.

Set API keys as environment variables before building, or let the app
prompt the user to paste a key into the Settings screen (see main.py).
"""

import os
import time
import requests


def fetch_binance(symbol: str, period: str = "1y", interval: str = "1d",
                   api_key: str = None) -> list:
    """
    Direct Binance public REST API — no API key needed for market data.
    symbol format: 'BTCUSDT', 'ETHUSDT', 'BNBUSDT' (no slash).
    """
    interval_map = {"1d": "1d", "1h": "1h", "1wk": "1w", "4h": "4h", "15m": "15m"}
    binance_interval = interval_map.get(interval, "1d")

    limit_map = {"1mo": 30, "3mo": 90, "6mo": 182, "1y": 365, "2y": 730}
    limit = min(limit_map.get(period, 365), 1000)   # Binance max 1000 per call

    params = {
        "symbol": symbol.upper().replace("/", "").replace("-", ""),
        "interval": binance_interval,
        "limit": limit,
    }
    resp = requests.get("https://api.binance.com/api/v3/klines", params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    if not isinstance(payload, list) or len(payload) == 0:
        raise ValueError(f"Binance error for '{symbol}': {payload}")

    # Each row: [open_time, open, high, low, close, volume, close_time, ...]
    return [float(row[4]) for row in payload]


def fetch_binance_ohlcv(symbol: str, period: str = "1y", interval: str = "1d") -> dict:
    """
    Full OHLCV from Binance (needed for RSI/MACD/Bollinger/ATR/Volume signals).
    Returns dict with 'close', 'high', 'low', 'volume' arrays (lists), oldest first.
    """
    interval_map = {"1d": "1d", "1h": "1h", "1wk": "1w", "4h": "4h", "15m": "15m"}
    binance_interval = interval_map.get(interval, "1d")
    limit_map = {"1mo": 30, "3mo": 90, "6mo": 182, "1y": 365, "2y": 730}
    limit = min(limit_map.get(period, 365), 1000)

    params = {
        "symbol": symbol.upper().replace("/", "").replace("-", ""),
        "interval": binance_interval,
        "limit": limit,
    }
    resp = requests.get("https://api.binance.com/api/v3/klines", params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, list) or len(payload) == 0:
        raise ValueError(f"Binance error for '{symbol}': {payload}")

    # row = [open_time, open, high, low, close, volume, close_time, ...]
    return {
        "close": [float(row[4]) for row in payload],
        "high": [float(row[2]) for row in payload],
        "low": [float(row[3]) for row in payload],
        "volume": [float(row[5]) for row in payload],
    }


def fetch_binance_funding_rate(symbol: str) -> dict:
    """
    Latest perpetual-futures funding rate for `symbol` (e.g. 'BTCUSDT').
    Positive rate = longs pay shorts (crowd leaning long); negative = the opposite.
    Public endpoint, no API key needed. Only meaningful for symbols with a
    USDT-M perpetual futures market on Binance.
    """
    symbol = symbol.upper().replace("/", "").replace("-", "")
    params = {"symbol": symbol, "limit": 1}
    resp = requests.get("https://fapi.binance.com/fapi/v1/fundingRate", params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    if not payload:
        raise ValueError(f"No funding rate data for '{symbol}' (may not have a futures market)")
    latest = payload[-1]
    return {"symbol": symbol, "funding_rate": float(latest["fundingRate"]),
            "funding_time": int(latest["fundingTime"])}


def fetch_binance_open_interest(symbol: str) -> dict:
    """
    Current open interest (number of outstanding futures contracts) for `symbol`.
    Rising OI + rising price = fresh money entering the trend (often bullish
    confirmation); falling OI + rising price can mean short covering. Public,
    no API key needed.
    """
    symbol = symbol.upper().replace("/", "").replace("-", "")
    resp = requests.get("https://fapi.binance.com/fapi/v1/openInterest",
                         params={"symbol": symbol}, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    if "openInterest" not in payload:
        raise ValueError(f"No open interest data for '{symbol}': {payload}")
    return {"symbol": symbol, "open_interest": float(payload["openInterest"])}


def fetch_binance_order_book_imbalance(symbol: str, depth: int = 100) -> dict:
    """
    Snapshot order-book imbalance: (bid_volume - ask_volume) / (bid_volume + ask_volume)
    over the top `depth` levels. Positive => more resting buy pressure nearby;
    negative => more resting sell pressure. Public, no API key needed.
    NOTE: this is a single noisy snapshot, not a time series — treat it as a
    rough, short-lived read on nearby order flow, not a standalone signal.
    """
    symbol = symbol.upper().replace("/", "").replace("-", "")
    resp = requests.get("https://api.binance.com/api/v3/depth",
                         params={"symbol": symbol, "limit": depth}, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    if "bids" not in payload or "asks" not in payload:
        raise ValueError(f"No order book data for '{symbol}': {payload}")

    bid_vol = sum(float(b[1]) for b in payload["bids"])
    ask_vol = sum(float(a[1]) for a in payload["asks"])
    total = bid_vol + ask_vol
    imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0
    return {"symbol": symbol, "bid_volume": bid_vol, "ask_volume": ask_vol,
            "imbalance": imbalance}


def fetch_fear_greed_index() -> dict:
    """
    Crypto Fear & Greed Index (alternative.me), 0 (extreme fear) - 100 (extreme
    greed). Market-wide sentiment, not asset-specific. Public, no API key needed.
    """
    resp = requests.get("https://api.alternative.me/fng/", params={"limit": 1}, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data")
    if not data:
        raise ValueError(f"Unexpected Fear & Greed response: {payload}")
    entry = data[0]
    return {"value": int(entry["value"]), "classification": entry["value_classification"]}


def fetch_twelvedata(symbol: str, period: str = "1y", interval: str = "1d",
                      api_key: str = None) -> list:
    api_key = api_key or os.environ.get("TWELVEDATA_API_KEY")
    if not api_key:
        raise RuntimeError("Twelve Data: missing API key")

    outputsize = {"1mo": 30, "3mo": 90, "6mo": 182, "1y": 365, "2y": 730}.get(period, 365)
    params = {
        "symbol": symbol,
        "interval": "1day" if interval == "1d" else interval,
        "outputsize": outputsize,
        "apikey": api_key,
    }
    resp = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    if "values" not in payload:
        raise ValueError(f"Twelve Data error: {payload}")

    values = list(reversed(payload["values"]))
    return [float(v["close"]) for v in values]


def fetch_alpha_vantage(symbol: str, period: str = "1y", interval: str = "1d",
                         api_key: str = None) -> list:
    api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("Alpha Vantage: missing API key")

    function = "TIME_SERIES_DAILY" if interval == "1d" else "TIME_SERIES_INTRADAY"
    params = {"function": function, "symbol": symbol, "apikey": api_key, "outputsize": "full"}
    if function == "TIME_SERIES_INTRADAY":
        params["interval"] = interval

    resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    series_key = next((k for k in payload if "Time Series" in k), None)
    if series_key is None:
        raise ValueError(f"Alpha Vantage error: {payload}")

    series = payload[series_key]
    dates_sorted = sorted(series.keys())
    return [float(series[d]["4. close"]) for d in dates_sorted]


def fetch_finnhub(symbol: str, period: str = "1y", interval: str = "1d",
                   api_key: str = None) -> list:
    api_key = api_key or os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        raise RuntimeError("Finnhub: missing API key")

    resolution = {"1d": "D", "1h": "60", "1wk": "W"}.get(interval, "D")
    days = {"1mo": 30, "3mo": 90, "6mo": 182, "1y": 365, "2y": 730}.get(period, 365)
    now = int(time.time())
    start = now - days * 86400

    params = {"symbol": symbol, "resolution": resolution, "from": start, "to": now, "token": api_key}
    resp = requests.get("https://finnhub.io/api/v1/stock/candle", params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("s") != "ok":
        raise ValueError(f"Finnhub error: {payload}")

    return [float(c) for c in payload["c"]]


PROVIDERS = {
    "Binance": fetch_binance,
    "Twelve Data": fetch_twelvedata,
    "Alpha Vantage": fetch_alpha_vantage,
    "Finnhub": fetch_finnhub,
}

# Providers that work without any API key
NO_KEY_REQUIRED = {"Binance"}


def fetch_with_fallback(symbol: str, period: str, interval: str, api_keys: dict,
                         order=None) -> list:
    """
    api_keys: dict like {"Twelve Data": "xxx", "Alpha Vantage": "yyy", ...}
    order: list of provider names to try, defaults to PROVIDERS insertion order.
    """
    order = order or list(PROVIDERS.keys())
    errors = {}
    for name in order:
        fn = PROVIDERS.get(name)
        if fn is None:
            continue
        key = api_keys.get(name)
        if name not in NO_KEY_REQUIRED and not key:
            errors[name] = "no API key configured"
            continue
        try:
            prices = fn(symbol, period=period, interval=interval, api_key=key)
            if prices:
                return prices, name
        except Exception as e:
            errors[name] = str(e)

    raise RuntimeError(f"All sources failed for '{symbol}': {errors}")
