"""
Core alpha extraction pipeline (Eq. 1-12 from the paper), pure numpy.
Same logic as the desktop alpha_extraction_bot.py, kept dependency-light
so it packages cleanly for Android via buildozer.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Technical indicators (pure numpy, no TA-Lib dependency — keeps APK light)
# ---------------------------------------------------------------------------

def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index, Wilder's smoothing. Returns array aligned to prices (NaN for warm-up)."""
    prices = np.asarray(prices, dtype=float)
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    out = np.full(len(prices), np.nan)
    if len(prices) <= period:
        return out

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    out[period] = 100 - (100 / (1 + rs))

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
        out[i + 1] = 100 - (100 / (1 + rs))

    return out


def ema(values: np.ndarray, span: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    alpha = 2 / (span + 1)
    out = np.full(len(values), np.nan)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram), all aligned to prices."""
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(prices: np.ndarray, period: int = 20, num_std: float = 2.0):
    """Returns (upper, mid, lower, percent_b). percent_b = (price - lower) / (upper - lower)."""
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    for i in range(period - 1, n):
        window = prices[i - period + 1 : i + 1]
        m = window.mean()
        s = window.std()
        mid[i] = m
        upper[i] = m + num_std * s
        lower[i] = m - num_std * s

    band_width = upper - lower
    percent_b = np.where(band_width != 0, (prices - lower) / band_width, 0.5)
    return upper, mid, lower, percent_b


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range, Wilder's smoothing."""
    highs, lows, closes = map(lambda a: np.asarray(a, dtype=float), (highs, lows, closes))
    n = len(closes)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    out = np.full(n, np.nan)
    if n <= period:
        return out
    out[period] = tr[1 : period + 1].mean()
    for i in range(period + 1, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


# ---------------------------------------------------------------------------
# Core pipeline (unchanged math from the paper, Eq. 1-12)
# ---------------------------------------------------------------------------

class AlphaExtractionBot:
    def __init__(self, returns: np.ndarray, d: int):
        self.R = np.asarray(returns, dtype=float)
        if self.R.ndim != 2:
            raise ValueError("returns must be a 2D array of shape (N, M)")
        self.N, self.M = self.R.shape
        if not (1 <= d <= self.M):
            raise ValueError(f"d must satisfy 1 <= d <= M (got d={d}, M={self.M})")
        self.d = d

        self.X = None
        self.sigma = None
        self.Y = None
        self.Lambda = None
        self.E = None
        self.E_norm = None
        self.beta = None
        self.epsilon = None
        self.w = None

    def compute_X(self):
        mean_R = self.R.mean(axis=1, keepdims=True)
        self.X = self.R - mean_R
        return self.X

    def compute_sigma(self):
        self.sigma = np.sqrt((self.X ** 2).mean(axis=1))
        return self.sigma

    def compute_Y(self):
        sigma_safe = np.where(self.sigma == 0, np.nan, self.sigma)
        Y_full = self.X / sigma_safe[:, None]
        self.Y = Y_full[:, : self.M - 1]
        return self.Y

    def compute_Lambda(self):
        cross_mean = np.nanmean(self.Y, axis=0, keepdims=True)
        Lambda_full = self.Y - cross_mean
        self.Lambda = Lambda_full[:, : self.M - 1]
        return self.Lambda

    def compute_E(self):
        window = self.R[:, self.M - self.d :]
        self.E = window.mean(axis=1)
        return self.E

    def compute_E_norm(self):
        sigma_safe = np.where(self.sigma == 0, np.nan, self.sigma)
        self.E_norm = self.E / sigma_safe
        return self.E_norm

    def orthogonalize(self):
        valid = ~np.isnan(self.E_norm) & ~np.isnan(self.Lambda).any(axis=1)
        A = self.Lambda[valid]
        y = self.E_norm[valid]

        A_design = np.hstack([np.ones((A.shape[0], 1)), A])
        beta_full, *_ = np.linalg.lstsq(A_design, y, rcond=None)
        self.beta = beta_full[1:]
        intercept = beta_full[0]

        y_hat = A_design @ beta_full
        resid = y - y_hat

        self.epsilon = np.full(self.N, np.nan)
        self.epsilon[valid] = resid
        return self.epsilon, self.beta, intercept

    def compute_weights(self):
        sigma_safe = np.where(self.sigma == 0, np.nan, self.sigma)
        raw_w = self.epsilon / sigma_safe
        raw_w = np.nan_to_num(raw_w, nan=0.0)

        abs_sum = np.sum(np.abs(raw_w))
        eta = 1.0 / abs_sum if abs_sum > 0 else 0.0
        self.w = eta * raw_w
        return self.w

    def combine(self, S: np.ndarray) -> float:
        S = np.asarray(S, dtype=float)
        if S.shape != (self.N,):
            raise ValueError(f"S must have shape ({self.N},)")
        return float(np.sum(self.w * S))

    def run(self, S: np.ndarray = None):
        self.compute_X()
        self.compute_sigma()
        self.compute_Y()
        self.compute_Lambda()
        self.compute_E()
        self.compute_E_norm()
        self.orthogonalize()
        self.compute_weights()

        result = {
            "sigma": self.sigma,
            "E": self.E,
            "E_norm": self.E_norm,
            "beta": self.beta,
            "epsilon": self.epsilon,
            "weights": self.w,
            "weight_abs_sum": float(np.sum(np.abs(self.w))),
        }
        if S is not None:
            result["combined_signal"] = self.combine(S)
        return result


DEFAULT_WINDOWS = [5, 10, 20, 40, 60, 90]


def build_signal_matrix(prices: np.ndarray, windows=DEFAULT_WINDOWS,
                         highs=None, lows=None, volumes=None):
    """
    Build R(i,s): the original multi-window momentum sub-signals, PLUS
    (when the extra data is provided) RSI, MACD histogram, Bollinger %B,
    and a volume z-score — each as its own time-aligned sub-signal so the
    same Eq. 1-12 pipeline can weigh them together.

    highs/lows/volumes are optional; pass them (same length as `prices`)
    to unlock ATR-derived risk info and the volume signal. RSI/MACD/BB only
    need `prices`.
    """
    max_w = max(windows)
    n_points = len(prices) - max_w
    if n_points < 20:
        raise ValueError("Not enough overlapping history across all lookback windows.")

    rows = []
    labels = []

    # --- momentum sub-signals (original design) ---
    for w in windows:
        p_now = prices[max_w:]
        p_then = prices[max_w - w : len(prices) - w]
        rows.append((p_now - p_then) / p_then)
        labels.append(f"{w}-bar momentum")

    # --- RSI, rescaled to a return-like signal centered at 0 ---
    rsi_vals = rsi(prices)[max_w:]
    rows.append(np.nan_to_num((rsi_vals - 50) / 50, nan=0.0))
    labels.append("RSI")

    # --- MACD histogram, normalized by price level so it's return-like ---
    _, _, hist = macd(prices)
    hist_norm = np.nan_to_num(hist[max_w:] / prices[max_w:], nan=0.0)
    rows.append(hist_norm)
    labels.append("MACD histogram")

    # --- Bollinger %B, rescaled to be centered at 0 ---
    _, _, _, percent_b = bollinger_bands(prices)
    rows.append(np.nan_to_num(percent_b[max_w:] - 0.5, nan=0.0))
    labels.append("Bollinger %B")

    # --- Volume z-score (only if volumes provided) ---
    if volumes is not None:
        volumes = np.asarray(volumes, dtype=float)
        vol_window = volumes[max_w:]
        roll_mean, roll_std = [], []
        for i in range(len(vol_window)):
            idx = max_w + i
            seg = volumes[max(0, idx - 20) : idx + 1]
            roll_mean.append(seg.mean())
            roll_std.append(seg.std() if seg.std() > 0 else 1.0)
        roll_mean, roll_std = np.array(roll_mean), np.array(roll_std)
        vol_z = (vol_window - roll_mean) / roll_std
        rows.append(np.clip(vol_z / 3, -1, 1))       # squashed to a comparable [-1,1] range
        labels.append("Volume z-score")

    R = np.vstack(rows)
    S_current = R[:, -1].copy()

    extras = {}
    if highs is not None and lows is not None:
        atr_vals = atr(highs, lows, prices)
        extras["atr_last"] = atr_vals[-1]
        extras["atr_series"] = atr_vals

    return R, S_current, labels, extras


def build_cross_sectional_matrix(price_dict: dict):
    """
    TRUE cross-sectional mode matching the paper's original design: N
    different assets (not sub-signals of one asset) are compared against
    each other. price_dict = {"BTCUSDT": [...], "ETHUSDT": [...], ...} —
    all price arrays should be aligned (same length, same timestamps).

    Returns R (N assets x M periods) of simple returns, S_current (latest
    return per asset), and the list of asset names in row order.
    """
    names = list(price_dict.keys())
    min_len = min(len(v) for v in price_dict.values())

    returns_rows = []
    for name in names:
        p = np.asarray(price_dict[name][-min_len:], dtype=float)
        r = np.diff(p) / p[:-1]
        returns_rows.append(r)

    R = np.vstack(returns_rows)
    S_current = R[:, -1].copy()
    return R, S_current, names


def simple_backtest(prices: np.ndarray, windows=DEFAULT_WINDOWS, d: int = 20,
                     rebalance_every: int = 5, lookback_bars: int = 250):
    """
    Rough historical check: repeatedly runs the pipeline on a trailing
    window, takes the resulting Combined Signal, and compares its SIGN to
    the asset's actual forward return over the next `rebalance_every` bars.

    This is a simple descriptive diagnostic, not a validated trading
    strategy or a guarantee of future performance.

    Returns a dict with hit_rate, avg_forward_return_when_long,
    avg_forward_return_when_short, and the list of (signal, forward_return)
    pairs for further inspection.
    """
    prices = np.asarray(prices, dtype=float)
    max_w = max(windows)
    min_needed = max_w + d + rebalance_every + 20
    if len(prices) < min_needed:
        raise ValueError(
            f"Need at least {min_needed} bars for a meaningful backtest, got {len(prices)}."
        )

    start = max(len(prices) - lookback_bars, max_w + d + 5)
    records = []

    for t in range(start, len(prices) - rebalance_every, rebalance_every):
        window_prices = prices[: t + 1]
        try:
            R, S_current, _, _ = build_signal_matrix(window_prices, windows=windows)
            bot = AlphaExtractionBot(R, d=min(d, R.shape[1] - 1))
            result = bot.run(S_current)
            signal = result["combined_signal"]
        except Exception:
            continue

        forward_return = (prices[t + rebalance_every] - prices[t]) / prices[t]
        records.append((signal, forward_return))

    if not records:
        raise ValueError("Backtest produced no valid samples over this history length.")

    signals = np.array([r[0] for r in records])
    fwd = np.array([r[1] for r in records])

    long_mask = signals > 0
    short_mask = signals < 0
    correct = ((signals > 0) & (fwd > 0)) | ((signals < 0) & (fwd < 0))

    return {
        "n_samples": len(records),
        "hit_rate": float(correct.mean()),
        "avg_forward_return_when_long": float(fwd[long_mask].mean()) if long_mask.any() else None,
        "avg_forward_return_when_short": float(fwd[short_mask].mean()) if short_mask.any() else None,
        "records": records,
    }
