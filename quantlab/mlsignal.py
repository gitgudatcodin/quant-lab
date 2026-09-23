"""Module 2 — ML Signal Backtester engine (pure Python port of engine.js).

Pure numpy port of the Quant Strategy Lab ML signal engine. No streamlit
imports — this module is UI-free and headless-testable.

DATA layout (from quantlab.data.load_ml_features):
    { months:[ym] (feature months), fnames:[20], symbols:[...],
      feats:{SYM:[ [20 numbers] per month | None ]},
      nextRet:{SYM:[bp int per month | None]}, spyRet:[bp int per month] }
Features at month f are known at month-end f; nextRet[f] is the realized
return from f -> f+1. A decision made at end of f earns nextRet[f].
Walk-forward: for each test year-block, train on the trailing W feature
months, then score the 12 feature months of the block. No look-ahead:
training labels at month m use nextRet[m], known by the end of the
training window (m <= train end). Test months never enter training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_ITERS = 250
DEFAULT_LR = 0.5


@dataclass
class Params:
    startYm: str = "2020-01"   # backtest start (first test year-block)
    trainWin: int = 36         # trailing training window, months
    lam: float = 0.10          # L2 regularization strength
    iters: int = DEFAULT_ITERS # batch GD iterations
    lr: float = DEFAULT_LR      # GD learning rate
    topFrac: float = 0.10      # fraction of ranked stocks held each month
    costBps: float = 10.0      # trading cost, basis points


def sigmoid(z):
    """Numerically stable logistic sigmoid (scalar or ndarray)."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def standardize(X):
    """Standardize columns using TRAIN rows only. Returns (mu, sd).

    Population std (ddof=0), mirroring the JS. Columns with zero (or
    non-finite) variance get sd=1.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    mu = X.mean(axis=0)
    sd = np.sqrt(((X - mu) ** 2).sum(axis=0) / n)
    sd[~(sd > 0)] = 1.0
    return mu, sd


def apply_z(X, mu, sd):
    """Apply training-set standardization to any matrix."""
    return (np.asarray(X, dtype=float) - mu) / sd


def train_logreg(X, y, lam, iters=DEFAULT_ITERS, lr=DEFAULT_LR):
    """Batch gradient descent, L2 logistic regression.

    X: standardized, no bias column — the intercept is appended here and is
    NOT penalized (mirrors engine.js). Returns w of length d+1, last = intercept.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    for _ in range(iters):
        err = sigmoid(Xb @ w) - y
        g = (Xb.T @ err) / n
        w[:d] -= lr * (g[:d] + lam * w[:d])
        w[d] -= lr * g[d]  # intercept: no penalty
    return w


def predict_proba(Xz, w):
    """Predicted P(label=1) for standardized rows."""
    Xz = np.asarray(Xz, dtype=float)
    w = np.asarray(w, dtype=float)
    return sigmoid(Xz @ w[:-1] + w[-1])


def _arrays(data):
    """Convert the JSON panel into numpy arrays (NaN = missing)."""
    symbols = list(data["symbols"])
    months = list(data["months"])
    fnames = list(data["fnames"])
    ns, nm, nf = len(symbols), len(months), len(fnames)
    feats = np.full((ns, nm, nf), np.nan)
    has = np.zeros((ns, nm), dtype=bool)
    nret = np.full((ns, nm), np.nan)
    for i, s in enumerate(symbols):
        fr = data["feats"][s]
        for t, row in enumerate(fr):
            if row is not None:
                feats[i, t, :] = row
                has[i, t] = True
        nr = data["nextRet"][s]
        for t, v in enumerate(nr):
            if v is not None:
                nret[i, t] = v
    spy = np.array([np.nan if v is None else v for v in data["spyRet"]], dtype=float)
    return symbols, months, fnames, feats, has, nret, spy


def build_train(feats, has, nret, train_idx):
    """Build one training matrix: rows = (symbol, train month).

    Label = 1 if the symbol's next-month return was >= the cross-sectional
    top-quintile cutoff that month, computed only inside the training window;
    months with < 50 valid returns are skipped. Mirrors engine.js.
    """
    Xs, ys = [], []
    for ti in train_idx:
        r = nret[:, ti]
        valid = ~np.isnan(r)
        if valid.sum() < 50:
            continue
        rv = np.sort(r[valid])
        cut = rv[int(len(rv) * 0.8)]  # top-quintile cutoff
        F = feats[:, ti, :]
        ok = valid & has[:, ti] & np.isfinite(F).all(axis=1)
        Xs.append(F[ok])
        ys.append((r[ok] >= cut).astype(float))
    if not Xs:
        return np.zeros((0, feats.shape[2])), np.zeros(0)
    return np.vstack(Xs), np.concatenate(ys)


class Runner:
    """Chunked walk-forward runner. step() trains + scores one year-block."""

    def __init__(self, data, params: Params):
        self.params = params
        (self.symbols, self.months, self.fnames,
         self.feats, self.has, self.next_ret, self.spy_ret) = _arrays(data)
        self.sym_index = {s: i for i, s in enumerate(self.symbols)}
        self.blocks = self._make_blocks()
        self.probs = {}  # test feature-month idx -> {sym: proba}
        self._imp = np.zeros(len(self.fnames))
        self._imp_n = 0
        self._bi = 0

    def _make_blocks(self):
        months = self.months
        P = self.params
        try:
            start_idx = months.index(P.startYm)
        except ValueError:
            start_idx = 0
        start_idx = max(0, start_idx)
        y0 = int(months[start_idx][:4])
        y_end = int(months[-1][:4])
        pos = {ym: i for i, ym in enumerate(months)}
        blocks = []
        for Y in range(y0, y_end + 1):
            test_idx = [pos[f"{Y}-{m:02d}"] for m in range(1, 13)
                        if f"{Y}-{m:02d}" in pos]
            if not test_idx:
                continue
            dec_prev = test_idx[0] - 1  # feature month before January
            train_idx = [ti for ti in range(dec_prev - P.trainWin + 1, dec_prev + 1)
                         if ti >= 0]
            if not train_idx:
                continue
            blocks.append({"trainIdx": train_idx, "testIdx": test_idx})
        return blocks

    @property
    def n_blocks(self):
        return len(self.blocks)

    @property
    def blocks_done(self):
        return self._bi

    @property
    def progress(self):
        return 1.0 if not self.blocks else self._bi / len(self.blocks)

    def done(self):
        return self._bi >= len(self.blocks)

    def step(self):
        """Train + score one year-block. Returns progress in [0, 1]."""
        if self.done():
            return 1.0
        b = self.blocks[self._bi]
        self._bi += 1
        X, y = build_train(self.feats, self.has, self.next_ret, b["trainIdx"])
        if len(X) > 10:
            mu, sd = standardize(X)
            w = train_logreg(apply_z(X, mu, sd), y,
                             self.params.lam, self.params.iters, self.params.lr)
            self._imp += np.abs(w[:-1])
            self._imp_n += 1
            for ti in b["testIdx"]:
                F = self.feats[:, ti, :]
                present = self.has[:, ti]
                Z = (F[present] - mu) / sd
                p = predict_proba(Z, w)
                syms = [self.symbols[i] for i in np.nonzero(present)[0]]
                self.probs[ti] = {s: float(v) for s, v in zip(syms, p)}
        return self.progress

    def importance(self):
        """Avg |coefficient| per feature across retrains, normalized to sum 1."""
        tot = self._imp.sum() or 1.0
        rows = [{"name": n, "w": float(v / tot)}
                for n, v in zip(self.fnames, self._imp)]
        rows.sort(key=lambda r: r["w"], reverse=True)
        return rows


def create_runner(data, params: Params) -> Runner:
    return Runner(data, params)


def walk_forward(data, params: Params, on_progress=None) -> Runner:
    """Run all year-blocks; on_progress(runner) called after each step."""
    r = create_runner(data, params)
    while not r.done():
        r.step()
        if on_progress is not None:
            on_progress(r)
    return r


def hold_ym(ym):
    """Holding month for a decision made at end of feature month ym = next month."""
    y, m = int(ym[:4]), int(ym[5:7]) + 1
    if m > 12:
        m, y = 1, y + 1
    return f"{y}-{m:02d}"


def backtest(runner: Runner, params: Params):
    """Backtest: each test month hold the top `topFrac` of symbols by predicted
    probability, equal-weighted, monthly rebalance. Costs on turnover.

    Mirrors engine.js exactly, including decile bookkeeping (decile 10 =
    highest predicted probability) and first-month full turnover cost.
    """
    cost = params.costBps / 10000.0
    test_idx = sorted(runner.probs.keys())
    n_top = max(1, round(len(runner.symbols) * params.topFrac))
    nr, spyv, months = runner.next_ret, runner.spy_ret, runner.months

    strat, ew, spy = [], [], []
    dec_s = np.zeros(10)
    dec_n = np.zeros(10, dtype=int)
    prev = None
    trades = 0

    for ti in test_idx:
        pm = runner.probs[ti]
        ranked = sorted(pm, key=pm.get, reverse=True)  # stable: ties keep sym order
        # decile analysis (decile 10 = highest predicted probability)
        per = max(1, len(ranked) // 10)
        for k, s in enumerate(ranked):
            dq = 9 - min(9, k // per)
            r = nr[runner.sym_index[s], ti]
            if not np.isnan(r):
                dec_s[dq] += r / 10000.0
                dec_n[dq] += 1
        top = ranked[:n_top]
        cur = set(top)
        if prev is not None:
            overlap = len(cur & prev)
            sell_w = (len(prev) - overlap) / len(prev)
            buy_w = (len(cur) - overlap) / len(cur)
            trades += (len(prev) - overlap) + (len(cur) - overlap)
        else:
            sell_w, buy_w = 0.0, 1.0
            trades += len(cur)
        rs = [nr[runner.sym_index[s], ti] / 10000.0 for s in top]
        rs = [v for v in rs if not np.isnan(v)]
        col = nr[:, ti]
        ok = ~np.isnan(col)
        gross = sum(rs) / len(rs) if rs else 0.0
        hym = hold_ym(months[ti])
        strat.append({"ym": hym, "r": float(gross - cost * (sell_w + buy_w))})
        ew.append({"ym": hym, "r": float(col[ok].mean() / 10000.0) if ok.any() else 0.0})
        spy.append({"ym": hym, "r": float(spyv[ti] / 10000.0)})
        prev = cur

    deciles = [{"decile": i + 1,
                "avgRet": float(dec_s[i] / dec_n[i]) if dec_n[i] else 0.0,
                "n": int(dec_n[i])} for i in range(10)]
    return {"strat": strat, "ew": ew, "spy": spy, "deciles": deciles,
            "trades": trades, "nMonths": len(test_idx)}


def stats(rets):
    """CAGR / Sharpe (population sd, monthly) / max drawdown of a return series."""
    rets = [float(r) for r in rets]
    n = len(rets)
    if not n:
        return {"cagr": 0.0, "sharpe": 0.0, "maxDD": 0.0}
    eq, peak, max_dd = 1.0, 1.0, 0.0
    s1 = s2 = 0.0
    for r in rets:
        eq *= 1.0 + r
        if eq > peak:
            peak = eq
        dd = eq / peak - 1.0
        if dd < max_dd:
            max_dd = dd
        s1 += r
        s2 += r * r
    mean = s1 / n
    sd = max(0.0, s2 / n - mean * mean) ** 0.5
    return {"cagr": eq ** (12.0 / n) - 1.0,
            "sharpe": (mean / sd) * (12.0 ** 0.5) if sd > 0 else 0.0,
            "maxDD": max_dd}


def yearly(legs):
    """legs: {name: [{ym, r}]} -> [{year, name: compounded return, ...}], sorted."""
    years = {}
    for name, pts in legs.items():
        for p in pts:
            y = p["ym"][:4]
            years.setdefault(y, {}).setdefault(name, []).append(p["r"])
    rows = []
    for y in sorted(years):
        row = {"year": y}
        for name in legs:
            rs = years[y].get(name, [])
            e = 1.0
            for r in rs:
                e *= 1.0 + r
            row[name] = e - 1.0
        rows.append(row)
    return rows
