"""Pairs-trading engine — pure Python port of quant-lab-build/pairs/engine.js.

No streamlit imports; numpy is the only dependency. All statistics are
computed point-in-time (causal windows only), mirroring the JS engine exactly:

- OLS hedge ratio, ADF(1 lag + constant) t-stat on gamma via normal-equations
  Gauss-Jordan (same singular-matrix bail: beta=zeros, se=inf).
- Half-life from AR(1) lambda, corr/dist prefilter on trailing 252 trading days.
- backtest_pair uses STRICTLY prior 60 days for rolling z (z_at excludes the
  current day), signals traded at close, position P&L = pos*0.5*(rA-rB).
- Costs: costFrac = costBps/10000 applied as v *= 1-costFrac on open AND close.
- Exits: |z|<=exitZ, |z|>=stopZ, or hold > maxHoldMult*hl (maxHold>=1).

API shape mirrors the JS (snake_case names):
    ols, adf, halflife, corr_range, dist_screen,
    Scanner (create_scanner), scan_pairs,
    backtest_pair, backtest_portfolio, stats, slice_data

DATA dict shape: {"dates":[str], "px":{SYM:[float]}, "spy":[float],
                  "sectors":{SYM:str}}.
"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP

import numpy as np

SCAN_W = 252   # trailing window for the pair scan
ROLL_W = 60    # rolling window for spread z-score (strictly prior days)
TRADING_DAYS = 252


# ---------------------------------------------------------------- stats ---

def _js_fixed(x: float, dec: int) -> float:
    """Emulate JS Number.toFixed(dec) then unary-plus (round-half-up)."""
    q = Decimal(1).scaleb(-dec)
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


def _js_round(x: float) -> int:
    """Emulate JS Math.round (round half up, toward +inf)."""
    return int(math.floor(x + 0.5))


def ols(x, y):
    """OLS: y = intercept + slope*x. Returns {"slope","intercept"}."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    sx, sy = x.sum(), y.sum()
    sxx, sxy = (x * x).sum(), (x * y).sum()
    den = n * sxx - sx * sx
    slope = 0.0 if den == 0 else (n * sxy - sx * sy) / den
    return {"slope": slope, "intercept": (sy - slope * sx) / (n or 1)}


def _ols_multi(X, y):
    """Multiple OLS via normal equations + Gauss-Jordan (mirrors the JS).

    X: (n,k) array whose first column is the constant. Returns (beta, se).
    Singular pivot (<1e-12) -> beta=zeros, se=inf (same bail as the JS).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    XtX = X.T @ X
    Xty = X.T @ y
    # augment with identity, Gauss-Jordan -> inverse (same loop order as JS)
    aug = np.hstack([XtX, np.eye(k)])
    for i in range(k):
        piv = aug[i, i]
        if abs(piv) < 1e-12:
            return np.zeros(k), np.full(k, np.inf)
        aug[i, :] /= piv
        for l in range(k):
            if l == i:
                continue
            f = aug[l, i]
            aug[l, :] -= f * aug[i, :]
    inv = aug[:, k:]
    beta = inv @ Xty
    resid = y - X @ beta
    rss = resid @ resid
    s2 = rss / (n - k) if n > k else 0.0
    se = np.sqrt(np.maximum(s2, 0.0) * np.maximum(np.diag(inv), 0.0))
    return beta, se


def adf(resid):
    """Augmented Dickey-Fuller (1 lag + constant) t-stat on gamma.

    Regression: dy_t = a + g*y_{t-1} + d*dy_{t-1} + e. More negative =>
    stronger evidence of stationarity (5% crit ~ -2.9).
    """
    resid = np.asarray(resid, dtype=float)
    n = resid.shape[0]
    if n < 10:
        return 0.0
    # t = 2..n-1: X rows [1, resid[t-1], resid[t-1]-resid[t-2]], y = dy
    prev = resid[1:-1]
    dprev = resid[1:-1] - resid[:-2]
    X = np.column_stack([np.ones(n - 2), prev, dprev])
    y = resid[2:] - resid[1:-1]
    beta, se = _ols_multi(X, y)
    g, seg = beta[1], se[1]
    if not math.isfinite(seg) or seg == 0:
        return 0.0
    return float(g / seg)


def halflife(resid):
    """Half-life of mean reversion from AR(1): dr_t = lambda*r_{t-1} + e."""
    resid = np.asarray(resid, dtype=float)
    prev = resid[:-1]
    dr = resid[1:] - resid[:-1]
    den = prev @ prev
    if den == 0:
        return math.inf
    lam = float((prev * dr).sum() / den)
    if lam >= -1e-9 or lam <= -1:
        return math.inf  # non-mean-reverting
    return -math.log(2) / math.log(1 + lam)


def corr_range(A, B, lo, hi):
    """Pearson correlation of A and B over index range [lo, hi)."""
    A = np.asarray(A, dtype=float)[lo:hi]
    B = np.asarray(B, dtype=float)[lo:hi]
    n = hi - lo
    sx, sy = A.sum(), B.sum()
    sxx, syy, sxy = (A * A).sum(), (B * B).sum(), (A * B).sum()
    den = math.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    return 0.0 if den == 0 else (n * sxy - sx * sy) / den


def dist_screen(A, B, lo, hi):
    """Distance screen: SSD of z-scored prices over [lo, hi). Population std."""
    A = np.asarray(A, dtype=float)[lo:hi]
    B = np.asarray(B, dtype=float)[lo:hi]
    n = hi - lo
    ma, mb = A.mean(), B.mean()
    sa = math.sqrt(((A - ma) ** 2).sum() / n)
    sb = math.sqrt(((B - mb) ** 2).sum() / n)
    if sa == 0 or sb == 0:
        return math.inf
    z = (A - ma) / sa - (B - mb) / sb
    return float((z * z).sum())


# --------------------------------------------------------------- scanner ---

class Scanner:
    """Chunked same-sector pair scanner (mirrors JS createScanner).

    Prefilter on trailing SCAN_W trading days: corr(prices) >= 0.7 AND
    distance-screen SSD <= 100. Survivors: OLS hedge ratio, ADF on residuals,
    half-life. Keep: adf < -2.9 AND half-life < 60 days.
    """

    def __init__(self, data):
        dates = data["dates"]
        self.n = len(dates)
        self.W = min(SCAN_W, self.n)
        self.lo = self.n - self.W
        syms = list(data["px"].keys())
        sectors = data.get("sectors", {})
        by_sec = {}
        for s in syms:
            by_sec.setdefault(sectors.get(s, "Unknown"), []).append(s)
        pairs = []
        for sec, g in by_sec.items():
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    pairs.append((g[i], g[j], sec))
        self.pairs = pairs
        self.px = {s: np.asarray(data["px"][s], dtype=float) for s in syms}
        self.idx = 0
        self.rows = []

    @property
    def total(self):
        return len(self.pairs)

    def _process_one(self, a, b, sec):
        lo, n, W = self.lo, self.n, self.W
        A = self.px[a][lo:n]
        B = self.px[b][lo:n]
        c = corr_range(A, B, 0, W)
        if not (c >= 0.7):
            return
        if not (dist_screen(A, B, 0, W) <= 100):
            return
        beta = ols(B, A)["slope"]
        resid = A - beta * B
        astat = adf(resid)
        if not (astat < -2.9):
            return
        hl = halflife(resid)
        if not (hl < 60):
            return
        self.rows.append({
            "a": a, "b": b, "sector": sec,
            "corr": _js_fixed(c, 3), "adf": _js_fixed(astat, 2),
            "hl": _js_fixed(hl, 1), "beta": _js_fixed(beta, 4),
        })

    def step(self, chunk=50):
        """Process the next `chunk` pairs. Returns status dict (like the JS)."""
        end = min(len(self.pairs), self.idx + chunk)
        while self.idx < end:
            a, b, sec = self.pairs[self.idx]
            self._process_one(a, b, sec)
            self.idx += 1
        return {"done": self.idx >= len(self.pairs), "scanned": self.idx,
                "total": len(self.pairs), "kept": len(self.rows),
                "rows": self.rows}


def create_scanner(data):
    return Scanner(data)


def scan_pairs(data, on_progress=None):
    """Synchronous convenience wrapper (mirrors JS scanPairs)."""
    sc = Scanner(data)
    while True:
        r = sc.step(400)
        if on_progress:
            on_progress(r["scanned"] / r["total"])
        if r["done"]:
            return r["rows"]


# -------------------------------------------------------------- backtests ---

def _rolling_z(spread, W=ROLL_W):
    """z[i] from the STRICTLY prior W days: (spread[i]-mean)/sd, else 0.

    Matches the JS zAt(i) which sums spread[lo..i) with lo = i-W.
    """
    n = spread.shape[0]
    z = np.zeros(n)
    c = np.concatenate([[0.0], np.cumsum(spread)])
    c2 = np.concatenate([[0.0], np.cumsum(spread * spread)])
    for i in range(W, n):
        m = (c[i] - c[i - W]) / W
        var = (c2[i] - c2[i - W]) / W - m * m
        sd = math.sqrt(var) if var > 0 else 0.0
        z[i] = (spread[i] - m) / sd if sd > 0 else 0.0
    return z


def backtest_pair(data, a, b, beta, P):
    """Backtest one pair. P keys: entryZ, exitZ, stopZ, maxHoldMult, costBps, hl."""
    A = np.asarray(data["px"][a], dtype=float)
    B = np.asarray(data["px"][b], dtype=float)
    dates = data["dates"]
    n = len(dates)
    W = ROLL_W
    hl = P.get("hl") or 30
    max_hold = max(1, _js_round((P.get("maxHoldMult") or 2) * hl))
    cost_frac = (P.get("costBps") or 0) / 10000.0  # per open and per close
    entry_z_th = P.get("entryZ", 2)
    exit_z_th = P.get("exitZ", 0.25)
    stop_z_th = P.get("stopZ", 4)

    spread = A - beta * B
    z = _rolling_z(spread, W)

    pos = 0  # +1 long spread, -1 short spread, 0 flat
    entry_day = 0
    entry_z = 0.0
    entry_v = 1.0
    entry_d = ""
    v = 1.0
    trades = []
    eq_d, eq_v = [], []

    for i in range(W, n):
        # overnight P&L earned by the position decided at yesterday's close
        rA = A[i] / A[i - 1] - 1.0
        rB = B[i] / B[i - 1] - 1.0
        v *= 1.0 + pos * 0.5 * (rA - rB)
        # signal + trade at today's close
        zi = z[i]
        if pos == 0:
            if zi <= -entry_z_th or zi >= entry_z_th:
                pos = 1 if zi <= -entry_z_th else -1
                entry_day = i
                entry_z = zi
                entry_d = dates[i]
                v *= 1.0 - cost_frac
                entry_v = v
        else:
            hold = i - entry_day
            if abs(zi) <= exit_z_th or abs(zi) >= stop_z_th or hold > max_hold:
                v *= 1.0 - cost_frac
                trades.append({
                    "a": a, "b": b,
                    "dir": "LONG spread" if pos == 1 else "SHORT spread",
                    "entryD": entry_d, "exitD": dates[i],
                    "entryZ": _js_fixed(entry_z, 2), "exitZ": _js_fixed(zi, 2),
                    "days": hold,
                    "retPct": _js_fixed((v / entry_v - 1.0) * 100.0, 2),
                })
                pos = 0
        eq_d.append(dates[i])
        eq_v.append(v)

    equity = [{"d": d, "v": vv} for d, vv in zip(eq_d, eq_v)]
    return {"a": a, "b": b, "equity": equity, "trades": trades,
            "stats": stats(equity, trades)}


def backtest_portfolio(data, pairs, N, P):
    """Top-N equal-capital portfolio. `pairs`: [{a,b,beta,hl}]. DATA = trade window."""
    top = list(pairs[:max(1, N)])
    w = 1.0 / len(top)
    per_pair = [backtest_pair(data, p["a"], p["b"], p["beta"],
                              {**P, "hl": p["hl"]}) for p in top]
    m = len(per_pair[0]["equity"])
    spy = np.asarray(data["spy"], dtype=float)
    spy0 = spy[len(spy) - m]
    equity, spy_eq = [], []
    for i in range(m):
        v = sum(w * pp["equity"][i]["v"] for pp in per_pair)
        equity.append({"d": per_pair[0]["equity"][i]["d"], "v": v})
        spy_eq.append({"d": per_pair[0]["equity"][i]["d"],
                       "v": float(spy[len(spy) - m + i] / spy0)})
    all_trades = []
    for pp in per_pair:
        all_trades.extend(pp["trades"])
    all_trades.sort(key=lambda t: t["exitD"])
    return {"equity": equity, "spy": spy_eq, "perPair": per_pair,
            "trades": all_trades, "stats": stats(equity, all_trades),
            "spyStats": stats(spy_eq, [])}


def stats(equity, trades=None):
    """Performance stats from a daily equity series [{"d","v"}]."""
    trades = trades or []
    n = len(equity)
    if n < 2:
        return {"cagr": 0, "sharpe": 0, "maxdd": 0, "win": 0, "n": len(trades)}
    v0, v1 = equity[0]["v"], equity[-1]["v"]
    rets = [equity[i]["v"] / equity[i - 1]["v"] - 1.0 for i in range(1, n)]
    mu = sum(rets) / len(rets)
    sd = math.sqrt(sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)) \
        if len(rets) > 1 else 0.0
    cagr = (v1 / v0) ** (TRADING_DAYS / (n - 1)) - 1.0 if v0 > 0 else 0.0
    sharpe = mu / sd * math.sqrt(TRADING_DAYS) if sd > 0 else 0.0
    peak = equity[0]["v"]
    dd = 0.0
    for e in equity:
        if e["v"] > peak:
            peak = e["v"]
        if peak > 0:
            dd = min(dd, e["v"] / peak - 1.0)
    wins = sum(1 for t in trades if t["retPct"] > 0)
    return {"cagr": _js_fixed(cagr, 4), "sharpe": _js_fixed(sharpe, 2),
            "maxdd": _js_fixed(dd, 4),
            "win": _js_fixed(wins / len(trades), 3) if trades else 0,
            "n": len(trades)}


def slice_data(data, d0, d1):
    """Slice DATA to an inclusive date window [d0, d1] (YYYY-MM-DD strings)."""
    idx = [i for i, d in enumerate(data["dates"]) if d0 <= d <= d1]
    return {
        "dates": [data["dates"][i] for i in idx],
        "px": {s: [data["px"][s][i] for i in idx] for s in data["px"]},
        "spy": [data["spy"][i] for i in idx],
        "sectors": data["sectors"],
    }
