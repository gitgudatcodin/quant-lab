"""RL trading agent — tabular Q-learning on SPY.

Faithful pure-Python port of ``quant-lab-build/rl/engine.js`` (plus the
training loop from ``template_ui.html``). No Streamlit / plotting imports:
this module is safe for headless testing and for ``st.cache_data``.

State (all point-in-time at day t):
  mom5  = sum of last 5 daily returns  -> 3 bins by train-period terciles
  rsi   = RSI(14) (Wilder)             -> 3 bins: <35 / 35-65 / >65
  vreg  = 20d vol / 60d vol            -> 2 bins by train-period median
  pos   = position held overnight into t (-1,0,+1) -> 3 bins
=> 54 states x 3 actions (target position -1,0,+1).
"""
from __future__ import annotations

import math
import random

ACTIONS = (-1, 0, 1)  # action index 0 -> short, 1 -> flat, 2 -> long

WARM = 61  # first usable day index (features need 60 days of history)


def date_idx(dates, ym):
    """Index of the last date <= ym (ym = "YYYY-MM-DD"). Binary search."""
    lo, hi, ans = 0, len(dates) - 1, 0
    while lo <= hi:
        mid = (lo + hi) >> 1
        if dates[mid] <= ym:
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ans


def _rolling_std(arr, t, w):
    m = 0.0
    for k in range(t - w + 1, t + 1):
        m += arr[k]
    m /= w
    v = 0.0
    for k in range(t - w + 1, t + 1):
        d = arr[k] - m
        v += d * d
    return math.sqrt(v / w)


def _wilder_rsi(rets):
    """Wilder RSI(14) exactly as in engine.js (seeded over days 1..14)."""
    n = len(rets)
    rsi = [50.0] * n
    g = l = 0.0
    for i in range(1, 15):
        d = rets[i]
        if d > 0:
            g += d
        else:
            l -= d
    g /= 14
    l /= 14
    rsi[14] = 100.0 if l < 1e-12 else 100.0 - 100.0 / (1.0 + g / l)
    for t in range(15, n):
        d = rets[t]
        g = (g * 13 + max(d, 0.0)) / 14
        l = (l * 13 + max(-d, 0.0)) / 14
        rsi[t] = 100.0 if l < 1e-12 else 100.0 - 100.0 / (1.0 + g / l)
    return rsi


def features(px):
    """Point-in-time features aligned to px index.

    Returns dict {rets, rsi, vol_reg, mom5, warm}. Index < warm is unusable.
    (engine.js also computed rNorm, but nothing consumes it, so it is omitted.)
    """
    px = list(px)
    n = len(px)
    rets = [0.0] * n
    for i in range(1, n):
        rets[i] = px[i] / px[i - 1] - 1.0
    rsi = _wilder_rsi(rets)
    vol_reg = [1.0] * n
    mom5 = [0.0] * n
    for t in range(60, n):
        v20 = _rolling_std(rets, t, 20)
        v60 = _rolling_std(rets, t, 60)
        vol_reg[t] = v20 / v60 if v60 > 1e-12 else 1.0
        mom5[t] = sum(rets[t - k] for k in range(5))
    return {"rets": rets, "rsi": rsi, "vol_reg": vol_reg, "mom5": mom5,
            "warm": WARM}


def _percentile(sorted_vals, p):
    """JS percentile(): index = min(n-1, max(0, floor(p*n)))."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    i = min(n - 1, max(0, math.floor(p * n)))
    return sorted_vals[i]


class Discretizer:
    """Tercile cutoffs / median computed on the TRAIN period only."""

    def __init__(self, F, i0, i1):
        ms = sorted(F["mom5"][t] for t in range(i0, i1 + 1))
        vs = sorted(F["vol_reg"][t] for t in range(i0, i1 + 1))
        self.m_lo = _percentile(ms, 1 / 3)
        self.m_hi = _percentile(ms, 2 / 3)
        self.v_med = _percentile(vs, 0.5)
        self._F = F

    def state(self, t, pos):
        F = self._F
        mom = F["mom5"][t]
        m = 0 if mom < self.m_lo else (2 if mom > self.m_hi else 1)
        r = F["rsi"][t]
        rb = 0 if r < 35 else (2 if r > 65 else 1)
        v = F["vol_reg"][t]
        vb = 0 if v <= self.v_med else 1
        p = pos + 1  # -1,0,+1 -> 0,1,2
        return ((m * 3 + rb) * 2 + vb) * 3 + p  # 0..53


def new_agent():
    """Fresh agent: 54x3 Q-table of zeros + per-state visit counts."""
    return {"Q": [0.0] * (54 * 3), "visits": [0] * 54}


def greedy_action(agent, s, tie_pos=None):
    """argmax Q(s,*); ties broken toward staying in the current position
    (1e-9 nudge -> less churn)."""
    best = -1e300
    ba = 1
    for a in range(3):
        tie = 1e-9 if (tie_pos is not None and ACTIONS[a] == tie_pos) else 0.0
        q = agent["Q"][s * 3 + a] + tie
        if q > best:
            best = q
            ba = a
    return ba


def run_episode(F, disc, agent, i0, i1, P, eps, rng, learn=True):
    """One full episode over train days [i0, i1) (i1 exclusive). Updates the
    agent in place. rng: random.Random instance (seeded by caller).

    Reward = target*rNext - cost - lambda*drawdown, where drawdown is the
    running peak-to-current equity fraction within the episode.
    Cost = cost_bps/10000 * |target - pos|.
    """
    cost_rate = P["cost_bps"] / 10000.0
    alpha, gamma, lam = P["alpha"], P["gamma"], P["lambda"]
    rets = F["rets"]
    Q, visits = agent["Q"], agent["visits"]
    pos = 0
    equity = 1.0
    peak = 1.0
    reward_sum = 0.0
    trades = 0
    for t in range(i0, i1):
        s = disc.state(t, pos)
        if learn and rng.random() < eps:
            a = int(rng.random() * 3)
        else:
            a = greedy_action(agent, s, pos)
        visits[s] += 1
        target = ACTIONS[a]
        change = abs(target - pos)
        cost = cost_rate * change
        if change > 0:
            trades += 1
        equity *= (1.0 - cost)
        r_next = rets[t + 1]
        equity *= (1.0 + target * r_next)
        if equity > peak:
            peak = equity
        dd = max(0.0, (peak - equity) / peak) if peak > 0 else 0.0
        reward = target * r_next - cost - lam * dd
        reward_sum += reward
        if learn:
            s2 = disc.state(t + 1, target)
            b = s2 * 3
            mx = Q[b]
            if Q[b + 1] > mx:
                mx = Q[b + 1]
            if Q[b + 2] > mx:
                mx = Q[b + 2]
            idx = s * 3 + a
            Q[idx] += alpha * (reward + gamma * mx - Q[idx])
        pos = target
    return {"reward": reward_sum, "trades": trades}


def stats(curve):
    """curve: list of (date, value). CAGR / Sharpe (daily, rf=0) / maxDD."""
    n = len(curve)
    if n < 2:
        return {"cagr": 0.0, "sharpe": 0.0, "maxDD": 0.0, "total": 0.0}
    rets = [curve[k][1] / curve[k - 1][1] - 1.0 for k in range(1, n)]
    m = sum(rets) / len(rets)
    sd = math.sqrt(sum((r - m) ** 2 for r in rets) / len(rets))
    v0, v1 = curve[0][1], curve[n - 1][1]
    total = v1 / v0 - 1.0
    cagr = (v1 / v0) ** (252.0 / (n - 1)) - 1.0
    peak = v0
    max_dd = 0.0
    for _, v in curve:
        if v > peak:
            peak = v
        dd = v / peak - 1.0
        if dd < max_dd:
            max_dd = dd
    return {"cagr": cagr,
            "sharpe": m / sd * math.sqrt(252.0) if sd > 0 else 0.0,
            "maxDD": max_dd, "total": total}


def evaluate(dates, F, disc, agent, j0, j1, P):
    """Greedy policy replayed on days [j0, j1] (inclusive). Returns equity
    curves (growth of $1), stats, trade count and action fractions."""
    cost_rate = P["cost_bps"] / 10000.0
    rets = F["rets"]
    eq, eq_bh = [], []
    act_count = [0, 0, 0]
    pos = 0
    equity = 1.0
    bh = 1.0
    trades = 0
    for t in range(j0, j1 + 1):
        # overnight into day t: yesterday's target position earns rets[t]
        equity *= (1.0 + pos * rets[t])
        bh *= (1.0 + rets[t])
        eq.append((dates[t], equity))
        eq_bh.append((dates[t], bh))
        # decide at close t (held into t+1); no new decision on the final day
        a = 1  # flat
        if t < j1:
            s = disc.state(t, pos)
            a = greedy_action(agent, s, pos)
        target = ACTIONS[a]
        change = abs(target - pos)
        if change > 0:
            trades += 1
        act_count[a] += 1
        equity *= (1.0 - cost_rate * change)
        pos = target
    n_days = j1 - j0 + 1
    return {
        "eq": eq, "eq_bh": eq_bh, "trades": trades,
        "action_frac": tuple(c / max(1, n_days) for c in act_count),
        "stats": stats(eq), "stats_bh": stats(eq_bh),
    }


def train_agent(F, disc, i0, i1, P, max_ep, seed,
                progress_cb=None, progress_every=5):
    """Full training loop (mirrors template_ui.html runTraining).

    epsilon decays linearly 1.0 -> 0.05 across episodes; early stop when the
    trailing-10-episode average reward stalls for 15 episodes (>= 25 total).
    Returns (agent, rewards, episodes_run, stopped_early).
    progress_cb(episodes_done, max_ep, rewards) is called every
    progress_every episodes and once at the end (UI progress hook; may be None).
    """
    rng = random.Random(seed)
    agent = new_agent()
    rewards = []
    best_avg = -math.inf
    stall = 0
    stopped_early = False
    denom = max(1, max_ep - 1)
    episodes_run = 0
    for k in range(max_ep):
        eps = 1.0 - 0.95 * (k / denom)
        r = run_episode(F, disc, agent, i0, i1, P, eps, rng, learn=True)
        rewards.append(r["reward"])
        episodes_run = k + 1
        k0 = max(0, len(rewards) - 10)
        avg = sum(rewards[k0:]) / (len(rewards) - k0)
        if len(rewards) >= 10:
            if avg > best_avg + 1e-9:
                best_avg = avg
                stall = 0
            else:
                stall += 1
            if stall >= 15 and len(rewards) >= 25:
                stopped_early = True
                break
        if progress_cb and (episodes_run % progress_every == 0):
            progress_cb(episodes_run, max_ep, rewards)
    if progress_cb:
        progress_cb(episodes_run, max_ep, rewards)
    return agent, rewards, episodes_run, stopped_early


_MOM_L = ["weak ↓", "mid", "strong ↑"]
_RSI_L = ["<35", "35–65", ">65"]
_VOL_L = ["calm", "hot"]
_POS_L = ["short", "flat", "long"]
_ACT_L = ["SHORT", "FLAT", "LONG"]


def q_peek(agent, disc, top_n=8):
    """Top states by visit count with the learned (greedy) action and Q row."""
    order = sorted(range(54), key=lambda s: agent["visits"][s], reverse=True)
    rows = []
    for s in order[:min(top_n, 54)]:
        p = s % 3
        v = (s // 3) % 2
        r = (s // 6) % 3
        m = s // 18
        a = greedy_action(agent, s, -99)
        b = s * 3
        rows.append({
            "state": s, "visits": agent["visits"][s],
            "mom": _MOM_L[m], "rsi": _RSI_L[r], "vol": _VOL_L[v],
            "pos": _POS_L[p], "action": _ACT_L[a],
            "q": [agent["Q"][b], agent["Q"][b + 1], agent["Q"][b + 2]],
        })
    return rows
