"""Module 1 — Pairs Trading (Streamlit port of quant-lab-build/pairs/template.html).

Scans S&P 500 stocks for cointegrated same-sector pairs, then backtests a
classic z-score mean-reversion strategy on their spread. All computation is
point-in-time; the engine lives in quantlab/pairs.py (pure Python, no streamlit).
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantlab import data as qd
from quantlab import pairs as E

st.set_page_config(page_title="Pairs Trading — Quant Strategy Lab",
                   page_icon="📊", layout="wide")

DATA = qd.load_pairs_daily()

# ------------------------------------------------------------- sidebar ---
st.sidebar.header("Strategy parameters")
entry_z = st.sidebar.slider("Z-score entry", 0.5, 3.0, 2.0, 0.25)
exit_z = st.sidebar.slider("Z-score exit", 0.0, 1.0, 0.25, 0.05)
stop_z = st.sidebar.slider("Z-score stop", 2.0, 6.0, 4.0, 0.5)
top_n = st.sidebar.slider("Top-N pairs", 1, 15, 5, 1)
cost_bps = st.sidebar.slider("Cost bps / leg / side", 0, 50, 10, 5)
max_hold_mult = st.sidebar.slider("Max-hold × half-life", 1.0, 4.0, 2.0, 0.5)

st.sidebar.header("Sample")
mode = st.sidebar.selectbox(
    "Sample",
    ["Train 2018–21 → test 2022–26", "Full sample 2018–26 (in-sample)"],
    index=0,
)
split = mode.startswith("Train")
train_d0 = st.sidebar.date_input("Train start", date(2018, 1, 1))
train_d1 = st.sidebar.date_input("Train end", date(2021, 12, 31))
test_d0 = st.sidebar.date_input("Test start", date(2022, 1, 1))
test_d1 = st.sidebar.date_input("Test end", date(2026, 8, 31))

P = {"entryZ": entry_z, "exitZ": exit_z, "stopZ": stop_z,
     "maxHoldMult": max_hold_mult, "costBps": cost_bps}


@st.cache_data(show_spinner=False)
def cached_scan(d0: str, d1: str, _cb=None):
    """Scan the [d0,d1] slice; keyed on scan params. Chunked for progress."""
    sl = E.slice_data(qd.load_pairs_daily(), d0, d1)
    sc = E.create_scanner(sl)
    while True:
        r = sc.step(200)
        if _cb is not None:
            _cb(r["scanned"] / r["total"])
        if r["done"]:
            return r["rows"]


st.sidebar.divider()
scan_clicked = st.sidebar.button("🔍 Scan for pairs", type="primary",
                                 width="stretch")
backtest_clicked = st.sidebar.button("📈 Backtest portfolio",
                                     width="stretch",
                                     disabled="pairs_rows" not in st.session_state)

# ------------------------------------------------------------------ scan ---
if scan_clicked:
    bar = st.progress(0.0, text="Scanning pairs…")
    def _cb(frac):  # noqa: E306 - progress callback for the cached scan
        bar.progress(frac, text=f"Scanning pairs… {frac:.0%}")
    rows = cached_scan(train_d0.isoformat(), train_d1.isoformat(), _cb=_cb)
    bar.empty()
    if not rows:
        st.warning("No pairs passed the screens on this window.")
        st.session_state.pop("pairs_rows", None)
    else:
        sl = E.slice_data(DATA, train_d0.isoformat(), train_d1.isoformat())
        with st.spinner(f"Computing in-sample Sharpe for {len(rows)} candidates…"):
            for r in rows:
                bt = E.backtest_pair(sl, r["a"], r["b"], r["beta"],
                                     {**P, "hl": r["hl"]})
                r["isSharpe"] = bt["stats"]["sharpe"]
        rows.sort(key=lambda r: -r["isSharpe"])
        st.session_state["pairs_rows"] = rows
        st.session_state["scan_P"] = dict(P)
        st.session_state["scan_window"] = (train_d0.isoformat(),
                                           train_d1.isoformat())
        st.success(f"Found {len(rows)} candidate pairs "
                   f"(train {train_d0} → {train_d1}).")

# ------------------------------------------------------------------ main ---
st.title("📊 Module 1 — Pairs Trading")
st.markdown(
    "Scans S&P 500 stocks for **cointegrated same-sector pairs**, then backtests "
    "a classic z-score mean-reversion strategy on their spread. "
    "Everything runs locally — no servers, no uploads."
)

st.header("Candidate pairs")
if "pairs_rows" not in st.session_state:
    st.info("Press **🔍 Scan for pairs** in the sidebar to start.")
else:
    rows = st.session_state["pairs_rows"]
    df = pd.DataFrame([{
        "Pair": f"{r['a']}/{r['b']}",
        "Sector": r["sector"],
        "Corr": r["corr"],
        "ADF": r["adf"],
        "Half-life d": r["hl"],
        "Hedge β": r["beta"],
        "IS Sharpe": round(r["isSharpe"], 2),
    } for r in rows])
    st.dataframe(df, width="stretch", hide_index=True)
    st.caption(
        "Sorted by in-sample Sharpe (train window). Screens on the trailing 252 "
        "trading days: corr ≥ 0.7, distance-screen SSD ≤ 100, ADF < −2.9, "
        "half-life < 60 d."
    )

st.header("Portfolio backtest")
if backtest_clicked and "pairs_rows" in st.session_state:
    rows = st.session_state["pairs_rows"]
    top = rows[:top_n]
    trade_slice = (E.slice_data(DATA, test_d0.isoformat(), test_d1.isoformat())
                   if split else E.slice_data(DATA, "2018-01-01", "2026-08-31"))
    window_label = (f"test {test_d0} → {test_d1} (out-of-sample)" if split
                    else "full sample 2018–26 (in-sample!)")
    with st.spinner(f"Backtesting top {len(top)} on {window_label}…"):
        pf = E.backtest_portfolio(trade_slice, top, len(top), P)
    st.subheader(f"Top-{len(top)} equal-weight · {window_label} · "
                 f"z {entry_z:g}/{exit_z:g}/{stop_z:g} · {cost_bps} bps")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[e["d"] for e in pf["equity"]],
                             y=[e["v"] for e in pf["equity"]],
                             name="Strategy", line=dict(color="#38bdf8")))
    fig.add_trace(go.Scatter(x=[e["d"] for e in pf["spy"]],
                             y=[e["v"] for e in pf["spy"]],
                             name="SPY buy & hold",
                             line=dict(color="#64748b", dash="dash")))
    fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.02),
                      yaxis_title="Growth of $1", xaxis_title="Date",
                      hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    s, ss = pf["stats"], pf["spyStats"]
    cols = st.columns(5)
    cols[0].metric("CAGR", f"{s['cagr']:.2%}", f"SPY {ss['cagr']:.2%}")
    cols[1].metric("Sharpe", f"{s['sharpe']:.2f}", f"SPY {ss['sharpe']:.2f}")
    cols[2].metric("Max drawdown", f"{s['maxdd']:.2%}", f"SPY {ss['maxdd']:.2%}")
    cols[3].metric("Win rate", f"{s['win']:.1%}")
    cols[4].metric("# trades", str(s["n"]))

    st.subheader("Trade log")
    trades = pf["trades"]
    if trades:
        tdf = pd.DataFrame([{
            "Pair": f"{t['a']}/{t['b']}", "Direction": t["dir"],
            "Entry": t["entryD"], "Exit": t["exitD"], "Days": t["days"],
            "Entry z": t["entryZ"], "Exit z": t["exitZ"], "Ret %": t["retPct"],
        } for t in trades])
        st.caption(f"Showing latest {min(400, len(tdf))} of {len(tdf)} trades.")
        st.dataframe(tdf.tail(400).iloc[::-1], width="stretch",
                     hide_index=True)
        st.download_button("⬇ Download trades CSV",
                           tdf.to_csv(index=False).encode(),
                           file_name="pairs_trades.csv", mime="text/csv")
    else:
        st.info("No trades in this window.")
elif "pairs_rows" not in st.session_state:
    pass
else:
    st.info("Adjust parameters, then press **📈 Backtest portfolio**.")

# -------------------------------------------------------------- explain ---
with st.expander("How it works", expanded=True):
    st.markdown("""
- **Cointegration:** two stocks (usually in the same business) wander up and down, but the *gap between them* tends to snap back — like two dogs on one leash. The Augmented Dickey-Fuller test checks whether that gap is truly mean-reverting rather than drifting apart forever.
- **Hedge ratio β:** from an OLS regression of stock A on stock B. The spread we trade is *A − β·B*, so a $1 move in B is offset by β dollars of A — that's what makes the position roughly market-neutral.
- **Z-score:** each day we measure how stretched the spread is versus its own rolling 60-day mean and standard deviation (strictly prior days — no look-ahead). z = +2 means the spread is 2 standard deviations wider than usual.
- **Trading rules:** when |z| ≥ entry (default 2), bet on snap-back — buy the cheap leg, short the expensive leg, equal dollars each side. Exit when |z| ≤ exit (0.25, spread normalized), stop out at |z| ≥ stop (4, the relationship may have broken), or after 2× the half-life in days (mean reversion is taking too long).
- **Dollar-neutral:** equal dollars long and short means broad market moves roughly cancel — profit comes from the two stocks converging, not from the market going up.
- **Why costs matter:** every round trip pays 10 bps per leg per side (4 charges). On a strategy that trades often for small gains, costs are frequently the difference between a good-looking and a real edge.
""")

with st.expander("Honest limitations (read before trusting it)"):
    st.markdown("""
- **Statistical arbitrage is hard.** Real-world stat-arb desks run with far more data, faster execution, and better borrow access than a daily-close backtest — and many still struggle. Treat this as education, not an edge.
- **Survivorship bias:** the universe is today's S&P 500 list. Companies that went bankrupt or were delisted are missing, which flatters results.
- **In-sample selection risk:** pairs are discovered and hedge ratios estimated on historical data; cointegration breaks in real life (mergers, regime shifts, fraud). Ranking pairs by in-sample Sharpe is itself a selection step that overfits.
- **Train/test split:** the default ranks pairs on 2018–21 and tests on 2022–26, which is more honest — but the scan rules and parameters were still chosen by a human who has seen markets.
- **Simplifications:** trades assumed at the daily close with no slippage beyond the bps haircut; no borrow fees or dividends on short positions; no corporate-action edge cases.
- **Strictly point-in-time:** every signal uses only data known at that day's close — rolling stats never peek ahead. That doesn't make it predictive.
- Backtested performance does not predict future results. This is an educational research tool, **not investment advice**.
""")
