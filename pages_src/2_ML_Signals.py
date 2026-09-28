"""Module 2 — ML Signal Backtester (Streamlit page).

A logistic-regression model learns which stocks land in next month's top
return quintile, retrained every year on trailing history (walk-forward, no
look-ahead). The portfolio holds each month's top-ranked stocks,
equal-weighted, monthly rebalanced.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantlab import data as qd
from quantlab import mlsignal as ml

st.title("🤖 ML Signal Backtester")
st.caption(
    "Quant Strategy Lab · Module 2. A logistic-regression model learns which stocks beat "
    "the market next month, retrained every year on trailing history (walk-forward, no "
    "look-ahead). The portfolio holds each month's top-ranked stocks, equal-weighted, "
    "monthly rebalanced. Everything runs locally — no data leaves this machine."
)

_tls = threading.local()  # per-thread progress sink for the cached run


@st.cache_data(show_spinner=False)
def _run_backtest(start_ym: str, train_win: int, lam: float,
                  top_pct: int, cost_bps: int):
    """Full walk-forward train + backtest. Cached on params; progress is
    reported through a thread-local sink set by the caller around this call."""
    t0 = time.perf_counter()
    DATA = qd.load_ml_features()
    P = ml.Params(startYm=start_ym, trainWin=train_win, lam=lam,
                  topFrac=top_pct / 100.0, costBps=float(cost_bps))

    def _on_progress(r):
        sink = getattr(_tls, "sink", None)
        if sink is not None:
            sink(r.progress, r.blocks_done, r.n_blocks)

    runner = ml.walk_forward(DATA, P, on_progress=_on_progress)
    bt = ml.backtest(runner, P)
    train_secs = time.perf_counter() - t0
    return {
        "strat": bt["strat"], "ew": bt["ew"], "spy": bt["spy"],
        "deciles": bt["deciles"], "trades": bt["trades"],
        "n_months": bt["nMonths"], "importance": runner.importance(),
        "train_secs": train_secs, "n_blocks": runner.n_blocks,
        "params": {"startYm": start_ym, "trainWin": train_win, "lambda": lam,
                   "topPct": top_pct, "costBps": cost_bps},
    }


def _equity(leg):
    xs, ys, e = [], [], 1.0
    for p in leg:
        e *= 1.0 + p["r"]
        xs.append(p["ym"])
        ys.append(e)
    return xs, ys


def _pct(v, d=2):
    return f"{v * 100:.{d}f}%"


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Model & portfolio")
    start_ym = st.selectbox(
        "Backtest start",
        ["2020-01", "2021-01", "2022-01", "2023-01", "2024-01"], index=0)
    train_win = st.selectbox(
        "Training window", [24, 36, 48], index=1,
        format_func=lambda m: f"{m} months")
    lam = st.slider("L2 regularization λ", 0.01, 1.00, 0.10, 0.01)
    top_pct = st.slider("Top fraction held", 5, 20, 10, 1, format="%d%%")
    cost_bps = st.slider("Trading cost", 0, 50, 10, 1, format="%d bps")
    run = st.button("▶ Train & backtest", type="primary", width="stretch")

# ---------------- run ----------------
if run:
    bar = st.progress(0.0)
    status = st.empty()
    status.text("Loading feature panel…")

    def _sink(p, done, total):
        bar.progress(p)
        status.text(f"Trained {done} / {total} yearly models…")

    _tls.sink = _sink
    try:
        wall0 = time.perf_counter()
        res = _run_backtest(start_ym, train_win, lam, top_pct, cost_bps)
        wall = time.perf_counter() - wall0
    finally:
        _tls.sink = None
    bar.progress(1.0)
    st.session_state["ml2_res"] = res
    st.session_state["ml2_key"] = (start_ym, train_win, lam, top_pct, cost_bps)
    st.session_state["ml2_cached"] = wall < 0.5 * max(res["train_secs"], 0.5)
    status.text(
        f"Done in {wall:.1f} s"
        + (" (served from cache)." if st.session_state["ml2_cached"] else ".")
        + " Adjust settings and re-run to compare."
    )

res = st.session_state.get("ml2_res")
if res is None:
    st.info("Set the model parameters in the sidebar, then press **▶ Train & backtest**. "
            "Training takes a few seconds on this machine.")
    st.stop()

P = res["params"]
legs = {"ML portfolio": res["strat"], "Equal-weight 500": res["ew"], "SPY buy & hold": res["spy"]}

# ---------------- equity chart ----------------
st.subheader("Growth of $1 — ML portfolio vs equal-weight universe vs SPY")
fig = go.Figure()
for name, color, leg in [("ML", "#38bdf8", res["strat"]),
                         ("EW500", "#94a3b8", res["ew"]),
                         ("SPY", "#34d399", res["spy"])]:
    x, y = _equity(leg)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name,
                             line=dict(color=color, width=2)))
fig.update_layout(height=380, hovermode="x unified",
                  yaxis_title="Value ($)", margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, width="stretch")

# ---------------- stat cards ----------------
cols = st.columns(3)
for col, (name, leg) in zip(cols, legs.items()):
    s = ml.stats([p["r"] for p in leg])
    with col:
        st.markdown(f"**{name}**")
        st.metric("CAGR", _pct(s["cagr"]))
        st.metric("Sharpe", f"{s['sharpe']:.2f}")
        st.metric("Max drawdown", _pct(s["maxDD"]))

# ---------------- decile spread ----------------
st.subheader("Decile spread — does the model actually rank stocks?")
st.caption("Avg next-month return by predicted-probability decile (10 = highest score).")
dec = sorted(res["deciles"], key=lambda d: d["decile"], reverse=True)
dec_df = pd.DataFrame([{
    "Decile": f"{d['decile']}" + (" ← highest score" if d["decile"] == 10 else ""),
    "Avg next-mo return": _pct(d["avgRet"]),
    "Stock-months": f"{d['n']:,}",
} for d in dec])
st.dataframe(dec_df, width="stretch", hide_index=True)
spread = res["deciles"][9]["avgRet"] - res["deciles"][0]["avgRet"]
if spread > 0.005:
    interp = "Upward slope — the model ranks stocks in this sample."
elif spread > 0:
    interp = "Nearly flat — weak or no ranking ability."
else:
    interp = "Downward — the model ranks backwards here."
st.caption(f"Top-minus-bottom decile spread: {_pct(spread)} per month. {interp} "
           f"{res['n_months']} months, walk-forward, no look-ahead.")

# ---------------- feature importance ----------------
st.subheader("Feature importance — avg |coefficient| across yearly retrains, standardized features")
imp = res["importance"]
fig2 = go.Figure(go.Bar(
    x=[x["w"] * 100 for x in reversed(imp)],
    y=[x["name"] for x in reversed(imp)],
    orientation="h",
    marker=dict(color="#38bdf8"),
    text=[f"{x['w'] * 100:.1f}%" for x in reversed(imp)],
    textposition="outside",
))
fig2.update_layout(height=max(300, 24 * len(imp)), xaxis_title="% of total |weight|",
                   margin=dict(l=10, r=60, t=10, b=40))
st.plotly_chart(fig2, width="stretch")

# ---------------- yearly table ----------------
st.subheader("Returns by year")
yr = ml.yearly({"ML": res["strat"], "EW500": res["ew"], "SPY": res["spy"]})
yr_df = pd.DataFrame([{
    "Year": r["year"],
    "ML portfolio": _pct(r["ML"]),
    "Equal-weight 500": _pct(r["EW500"]),
    "SPY": _pct(r["SPY"]),
    "Excess vs SPY": _pct(r["ML"] - r["SPY"]),
} for r in yr])
st.dataframe(yr_df, width="stretch", hide_index=True)

# ---------------- trades ----------------
st.subheader("Trades")
avg_mo = res["trades"] / max(1, res["n_months"])
cache_note = "served from cache" if st.session_state.get("ml2_cached") else "freshly trained"
st.markdown(
    f"One-way trades: **{res['trades']:,}** over {res['n_months']} months "
    f"(avg {avg_mo:.0f}/mo; buys + sells counted separately). Cost model: {P['costBps']} bps applied to buys + sells on turnover. "
    f"Walk-forward training took **{res['train_secs']:.1f} s** "
    f"({res['n_blocks']} yearly models, 250 GD iters, {P['trainWin']}-mo windows, "
    f"λ={P['lambda']}) — {cache_note}."
)

# ---------------- honest caveats ----------------
st.subheader("How this backtest works & its limits")
with st.expander("How it works (plain language)", expanded=True):
    st.markdown(
        "**Logistic regression = a weighted score turned into a probability.** Each month, every "
        "stock gets 20 numbers: 4 price features (12-month momentum, 60-day change, distance from "
        "52-week high, 12-month volatility) and 8 fundamental features (revenue growth, profitability "
        "flags, margin change, leverage, quarterly growth, accruals, gross profitability) plus 8 flags "
        "marking where fundamentals are missing. The model learns one weight per feature; the weighted "
        "sum is squashed into a probability that the stock lands in next month's top quintile of returns.\n\n"
        "**Walk-forward = retrain the way you'd do it live.** For each test year, the model trains only "
        "on the trailing 24/36/48 months, then scores the next 12 months. Training labels use next-month "
        "returns that were already known inside the training window; test months never appear in training. "
        "There is no look-ahead.\n\n"
        "**The portfolio** holds the top 10% (adjustable 5–20%) of stocks by predicted probability, "
        "equal-weighted, rebalanced monthly, with a per-trade cost applied to turnover."
    )
with st.expander("Read this before trusting the numbers"):
    st.markdown(
        "- **Monthly stock-picking with ML is hard.** The training labels (top-quintile next-month "
        "return) are extremely noisy — most of next month's return is luck, not signal. A model can "
        "find patterns in noise that don't repeat.\n"
        "- **A small edge vanishes after costs.** Try raising the cost slider to 20–50 bps: realistic "
        "for a retail account turning over the portfolio monthly. Watch the excess return shrink.\n"
        "- **Survivorship bias.** The universe is today's S&P 500 list — stocks that were delisted or "
        "dropped from the index along the way are missing, which flatters every backtest on this data.\n"
        "- **Regime dependence.** 2020–2026 was dominated by mega-cap momentum. A model that learned "
        "\"buy what's going up\" looks brilliant here and can fail for years when the regime changes.\n"
        "- **Backtest ≠ future.** These results describe one specific history, not what the model will "
        "do next month. This is a learning lab, not investment advice."
    )
st.warning(
    "⚠ **Honest framing:** if the decile table above is flat (decile 10 ≈ decile 1), the model is not "
    "ranking stocks — any portfolio outperformance is luck or factor exposure, not stock selection. "
    "If it slopes upward, treat it as *a* signal in *a* regime, and demand it survives higher costs "
    "before believing it."
)
