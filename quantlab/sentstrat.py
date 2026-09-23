"""Quant Strategy Lab — Sentiment strategy engine (Module 5).

Faithful Python port of quant-lab-build/sentiment-strategy/engine.js.
PURE LOGIC: no streamlit, no DOM, no network.

Strategy: post-earnings-announcement drift on LM-lexicon tone.
- Each earnings event is scored from its press-release text (score known at filing).
- At entry (next trading day after filing), the event's score is ranked against
  the trailing 252 trading days of scores (strictly point-in-time -- NEVER
  ranked within a calendar quarter, which would be look-ahead).
- Top `frac` of the trailing distribution -> long candidate; bottom `frac` -> short.
- Hold exactly H trading days; equal-weight; dollar-neutral in long/short mode.
- 10 bps cost per entry and per exit leg.

DATA = { dates:[...], px:{SYM:[...]}, spy:[...], sectors:{SYM:..},
         events:[{t,fd,ed,s,pos,neg,unc,lit,w}] }
  (events.json actually uses short keys t/fd/ed/s; long aliases
   ticker/fileDate/entryDate/score are accepted too.)
opts = { hold:30, mode:'ls'|'long', frac:0.333, maxPos:30, cost:0.001 }
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

_EPS = 1e-12


def bsearch(dates: Sequence[str], d: str) -> int:
    """Binary search for exact date string; -1 if absent."""
    lo, hi = 0, len(dates) - 1
    while lo <= hi:
        m = (lo + hi) >> 1
        x = dates[m]
        if x == d:
            return m
        if x < d:
            lo = m + 1
        else:
            hi = m - 1
    return -1


def px_ret(px: Dict[str, Sequence[Optional[float]]], t: str, i: int) -> float:
    a = px[t][i - 1]
    b = px[t][i]
    if a is None or b is None or a <= 0:
        return 0.0
    return b / a - 1.0


def _ev_get(e: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in e:
            return e[k]
    return default


def _normalize_event(e: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "t": _ev_get(e, "t", "ticker"),
        "fd": _ev_get(e, "fd", "fileDate"),
        "ed": _ev_get(e, "ed", "entryDate"),
        "s": _ev_get(e, "s", "score"),
        "pos": _ev_get(e, "pos"),
        "neg": _ev_get(e, "neg"),
        "unc": _ev_get(e, "unc"),
        "lit": _ev_get(e, "lit"),
        "w": _ev_get(e, "w", "total"),
    }


def assign_sides(evs: List[Dict[str, Any]], frac: float) -> None:
    """Assign each event a long/short/neutral side using only information
    known at its entry date: its score's percentile vs the trailing 252
    trading days of event scores (expanding window when fewer than 20 prior
    events exist). Mutates evs in place; `evs` must be sorted by ei."""
    head = 0
    window: List[float] = []  # scores of events with ei < current, in ei order
    n = len(evs)
    for e in evs:
        while head < n and evs[head]["ei"] < e["ei"]:
            window.append(evs[head]["s"])
            head += 1
        # drop scores older than 252 trading days before e.ei
        cutoff = e["ei"] - 252
        # window[j] belongs to evs[j] (built iterating evs in ei order)
        pool = [s for j, s in enumerate(window) if evs[j]["ei"] >= cutoff]
        if len(pool) >= 20:
            below = sum(1 for s in pool if s < e["s"])
            pct = below / len(pool)
        else:
            # early-sample fallback: expanding pool, else score sign
            if len(window) >= 5:
                below = sum(1 for s in window if s < e["s"])
                pct = below / len(window)
            else:
                pct = 0.99 if e["s"] > 0.05 else (0.01 if e["s"] < -0.05 else 0.5)
        e["side"] = 1 if pct >= 1 - frac else (-1 if pct <= frac else 0)
        e["poolN"] = len(pool)


def summarize(rets: Sequence[float]) -> Dict[str, Any]:
    """rets: list of daily returns (fraction). Returns curve + stats."""
    eq = 1.0
    peak = 1.0
    maxdd = 0.0
    s = 0.0
    s2 = 0.0
    curve = [1.0]
    n = len(rets)
    for r in rets:
        eq *= 1.0 + r
        curve.append(eq)
        if eq > peak:
            peak = eq
        dd = eq / peak - 1.0
        if dd < maxdd:
            maxdd = dd
        s += r
        s2 += r * r
    yrs = n / 252.0
    mean = s / n if n else 0.0
    sd = math.sqrt(max(s2 / max(n, 1) - mean * mean, _EPS))
    return {
        "curve": curve,
        "final": eq,
        "cagr": (max(eq, 1e-9) ** (1.0 / yrs) - 1.0) if yrs > 0 else 0.0,
        "sharpe": (mean / sd * math.sqrt(252.0)) if sd > 0 else 0.0,
        "maxdd": maxdd,
        "n": n,
    }


def _comp(arr: Sequence[float]) -> float:
    e = 1.0
    for r in arr:
        e *= 1.0 + r
    return e - 1.0


def run_backtest(DATA: Dict[str, Any], opts: Dict[str, Any]) -> Dict[str, Any]:
    dates: List[str] = DATA["dates"]
    px: Dict[str, Sequence[Optional[float]]] = DATA["px"]
    spy: Sequence[float] = DATA["spy"]
    n = len(dates)
    H = int(opts["hold"])
    cost = float(opts["cost"])
    frac = float(opts["frac"])
    max_pos = int(opts["maxPos"])
    mode = opts["mode"]

    evs: List[Dict[str, Any]] = []
    for raw in DATA["events"]:
        e = _normalize_event(raw)
        ei = bsearch(dates, e["ed"])
        if ei < 0 or ei + H >= n or e["t"] not in px:
            continue
        e["ei"] = ei
        e["sec"] = (DATA.get("sectors") or {}).get(e["t"], "")
        evs.append(e)
    evs.sort(key=lambda a: a["ei"])  # stable, mirrors JS sort on ei
    assign_sides(evs, frac)

    rets = [0.0] * n
    active: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    add_ptr = 0
    sum_active = 0.0
    sum_gross = 0.0
    cost_tot = 0.0

    for i in range(1, n):
        # enter at yesterday's close: events with ei === i-1 become active today
        while add_ptr < len(evs) and evs[add_ptr]["ei"] == i - 1:
            e = evs[add_ptr]
            if e["side"] != 0:
                active.append({"ev": e, "side": e["side"], "lastW": 0.0})
            add_ptr += 1
        # exit at yesterday's close: events with ei+H === i-1 leave
        for k in range(len(active) - 1, -1, -1):
            a = active[k]
            if a["ev"]["ei"] + H == i - 1:
                dc = cost * abs(a["lastW"])
                cost_tot += dc
                hret = a["side"] * (px[a["ev"]["t"]][a["ev"]["ei"] + H]
                                    / px[a["ev"]["t"]][a["ev"]["ei"]] - 1.0)
                trades.append({
                    "t": a["ev"]["t"], "fd": a["ev"]["fd"], "ed": a["ev"]["ed"],
                    "side": a["ev"]["side"], "s": a["ev"]["s"],
                    "hret": hret, "exit": dates[a["ev"]["ei"] + H],
                    "sec": a["ev"]["sec"],
                })
                del active[k]
        longs = sorted((a for a in active if a["side"] > 0),
                       key=lambda a: a["ev"]["s"], reverse=True)[:max_pos]
        shorts = sorted((a for a in active if a["side"] < 0),
                        key=lambda a: a["ev"]["s"])[:max_pos]
        tL = (1.0 if mode == "long" else 0.5) if longs else 0.0
        tS = 0.5 if (mode == "ls" and shorts) else 0.0
        dr = 0.0
        dc = 0.0
        for a in longs:
            w = tL / len(longs)
            a["lastW"] = w
            dr += w * px_ret(px, a["ev"]["t"], i)
            if a["ev"]["ei"] == i - 1:
                dc += cost * w
                cost_tot += cost * w
        for a in shorts:
            w = tS / len(shorts)
            a["lastW"] = -w
            dr += -w * px_ret(px, a["ev"]["t"], i)
            if a["ev"]["ei"] == i - 1:
                dc += cost * w
                cost_tot += cost * w
        rets[i] = dr - dc
        sum_active += len(active)
        sum_gross += tL + tS

    strat = summarize(rets[1:])
    spy_r = [spy[i] / spy[i - 1] - 1.0 for i in range(1, n)]
    bench = summarize(spy_r)

    # Event study: average cumulative return by score quintile, days 0..60
    by_score = sorted(evs, key=lambda a: a["s"])
    qn = max(1, len(by_score) // 5)
    drift = []
    for q in range(5):
        grp = by_score[q * qn:] if q == 4 else by_score[q * qn:(q + 1) * qn]
        cum = [0.0] * 61
        cnt = [0] * 61
        for e in grp:
            p0 = px[e["t"]][e["ei"]]
            if not p0:
                continue
            for d in range(61):
                if e["ei"] + d >= n:
                    break
                pd = px[e["t"]][e["ei"] + d]
                if pd is None:
                    continue
                cum[d] += pd / p0 - 1.0
                cnt[d] += 1
        drift.append({
            "q": q,
            "n": len(grp),
            "lo": grp[0]["s"] if grp else 0.0,
            "hi": grp[-1]["s"] if grp else 0.0,
            "cum": [c / cnt[d] if cnt[d] else None for d, c in enumerate(cum)],
        })

    # Monthly and yearly PnL
    monthly: Dict[str, List[float]] = {}
    yearly: Dict[str, List[float]] = {}
    for i in range(1, n):
        mk = dates[i][:7]
        yk = dates[i][:4]
        monthly.setdefault(mk, []).append(rets[i])
        yearly.setdefault(yk, []).append(rets[i])
    monthly_arr = [{"m": k, "r": _comp(monthly[k])} for k in sorted(monthly)]
    yearly_arr = []
    for k in sorted(yearly):
        s = summarize(yearly[k])
        yearly_arr.append({"y": k, "r": s["final"] - 1.0,
                           "sharpe": s["sharpe"], "maxdd": s["maxdd"],
                           "n": s["n"]})

    # Win rate / avg holding return over closed positions
    wins = sum(1 for t in trades if t["hret"] > 0)
    sum_h = sum(t["hret"] for t in trades)

    def _fwd(e: Dict[str, Any]) -> float:
        return px[e["t"]][e["ei"] + H] / px[e["t"]][e["ei"]] - 1.0

    def _ev_row(e: Dict[str, Any], with_side: bool = True) -> Dict[str, Any]:
        row = {"t": e["t"], "fd": e["fd"], "ed": e["ed"], "s": e["s"],
               "fwd": _fwd(e), "sec": e["sec"]}
        if with_side:
            row["side"] = e["side"]
        return row

    recent = [_ev_row(e) for e in reversed(evs[-40:])]
    top_s = [_ev_row(e, with_side=False)
             for e in sorted(evs, key=lambda a: a["s"], reverse=True)[:10]]
    bot_s = [_ev_row(e, with_side=False)
             for e in sorted(evs, key=lambda a: a["s"])[:10]]

    yrs = (n - 1) / 252.0
    return {
        "dates": dates,
        "rets": rets,
        "strat": strat,
        "bench": bench,
        "drift": drift,
        "monthly": monthly_arr,
        "yearly": yearly_arr,
        "trades": trades,
        "recent": recent,
        "topS": top_s,
        "botS": bot_s,
        "nEvents": len(evs),
        "nLong": sum(1 for e in evs if e["side"] > 0),
        "nShort": sum(1 for e in evs if e["side"] < 0),
        "winRate": wins / len(trades) if trades else 0.0,
        "avgHold": sum_h / len(trades) if trades else 0.0,
        # turnover: round-trip notional / yr
        "turnover": cost_tot / (cost if cost else 1e-9) / yrs if yrs > 0 else 0.0,
        "avgActive": sum_active / max(n - 1, 1),
        "avgGross": sum_gross / max(n - 1, 1),
        "hold": H,
        "mode": mode,
    }
