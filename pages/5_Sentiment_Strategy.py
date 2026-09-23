"""Module 5 — Sentiment Trading Strategy.

Post-earnings-announcement drift traded on Loughran-McDonald press-release
tone. Each event is ranked against the TRAILING 12 months of scores known at
its entry date (strictly point-in-time -- never ranked within a calendar
quarter, which would be look-ahead).

Honest headline: with defaults the long/short version LOSES money. The long
leg works (+2.4%/holding, ~56% win); the short leg fails (-3.6%, ~40% win) --
so long-only beats SPY.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantlab import data as qd
from quantlab import sentstrat

st.set_page_config(page_title="Sentiment Trading Strategy", layout="wide")
st.title("📰 Sentiment Trading Strategy")
st.markdown(
    "Post-earnings-announcement drift, traded on tone. Every quarter, companies "
    "file earnings press releases (8-K item 2.02). This lab scores each release "
    "with the Loughran-McDonald financial sentiment lexicon, goes **long the most "
    "positive-tone names and short the most negative-tone names**, and holds for a "
    "fixed window. Everything runs locally — no servers, no uploads."
)


@st.cache_data(show_spinner="Running sentiment backtest…")
def _run_backtest(hold: int, mode: str, frac: float, max_pos: int, cost_bps: int):
    px100 = qd.load_sentstrat_prices()
    events = qd.load_events()
    data = {
        "dates": px100["dates"],
        "px": px100["px"],
        "spy": px100["spy"],
        "sectors": px100.get("sectors", {}),
        "events": events,
    }
    return sentstrat.run_backtest(
        data,
        {"hold": hold, "mode": mode, "frac": frac,
         "maxPos": max_pos, "cost": cost_bps / 10000.0},
    )


# ---------------- sidebar ----------------
st.sidebar.header("Controls")
hold = st.sidebar.slider("Holding period (trading days)", 10, 60, 30, 5)
frac_pct = st.sidebar.slider("Sentiment tail (tercile fraction)", 20, 50, 33, 5,
                             format="%d%%")
frac = frac_pct / 100.0
max_pos = st.sidebar.slider("Max positions / side", 5, 60, 30, 5)
cost_bps = st.sidebar.slider("Cost (bps / leg)", 0, 50, 10, 5)
mode = st.sidebar.selectbox(
    "Mode",
    options=["ls", "long"],
    format_func=lambda m: "Long / short (dollar-neutral)" if m == "ls" else "Long-only",
    index=0,
)

px100 = qd.load_sentstrat_prices()
events = qd.load_events()
st.sidebar.caption(
    f"**{len(events):,}** scored earnings events · "
    f"{len(px100['px'])} tickers · {px100['dates'][0]} → {px100['dates'][-1]}\n\n"
    "8-K item 2.02 press releases, SEC EDGAR"
)

res = _run_backtest(hold, mode, frac, max_pos, cost_bps)
s, b = res["strat"], res["bench"]


def fmt_p(x, d=1):
    return f"{'+' if x >= 0 else ''}{x * 100:.{d}f}%"


# ---------------- honest headline ----------------
longs = [t for t in res["trades"] if t["side"] > 0]
shorts = [t for t in res["trades"] if t["side"] < 0]
la = np.mean([t["hret"] for t in longs]) if longs else 0.0
lw = np.mean([t["hret"] > 0 for t in longs]) if longs else 0.0
sa = np.mean([t["hret"] for t in shorts]) if shorts else 0.0
sw = np.mean([t["hret"] > 0 for t in shorts]) if shorts else 0.0

if mode == "ls":
    st.warning(
        f"**Honest headline:** with these settings the long/short version **loses money** "
        f"(CAGR {fmt_p(s['cagr'])} vs SPY {fmt_p(b['cagr'])}). The long leg works "
        f"({fmt_p(la)}/holding, {lw * 100:.0f}% win); the short leg fails "
        f"({fmt_p(sa)}, {sw * 100:.0f}% win). Switch to **long-only** to see the "
        "side of the trade that actually works."
    )
else:
    st.success(
        f"**Honest headline:** long-only CAGR {fmt_p(s['cagr'])} vs SPY "
        f"{fmt_p(b['cagr'])}, Sharpe {s['sharpe']:.2f} vs {b['sharpe']:.2f}. "
        f"The tone signal works on the long side ({fmt_p(la)}/holding, "
        f"{lw * 100:.0f}% win); the short leg ({fmt_p(sa)}, {sw * 100:.0f}% win) "
        "is what drags the L/S version underwater."
    )

st.subheader(
    f"Portfolio backtest — {res['nEvents']:,} events · {res['nLong']} long / "
    f"{res['nShort']} short candidates · {'L/S' if mode == 'ls' else 'long-only'} · H={hold}d"
)

# ---------------- stat cards ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("CAGR", fmt_p(s["cagr"]), f"SPY {fmt_p(b['cagr'])}",
          delta_color="normal")
c2.metric("Sharpe", f"{s['sharpe']:.2f}", f"SPY {b['sharpe']:.2f}")
c3.metric("Max drawdown", fmt_p(s["maxdd"]), f"SPY {fmt_p(b['maxdd'])}")
c4.metric("Win rate", f"{res['winRate'] * 100:.1f}%",
          f"{len(res['trades']):,} closed positions")
c5, c6, c7, c8 = st.columns(4)
c5.metric("Avg holding ret", fmt_p(res["avgHold"]), "per position, signed")
c6.metric("Turnover", f"{res['turnover']:.1f}×/yr", "one-way notional")
c7.metric("Events", f"{res['nEvents']:,}",
          f"{res['nLong']} long / {res['nShort']} short")
c8.metric("Avg exposure", fmt_p(res["avgGross"], 0),
          f"avg {res['avgActive']:.1f} positions")

# leg breakdown (only meaningful in L/S, but harmless in long-only)
if mode == "ls":
    d1, d2 = st.columns(2)
    d1.metric("Long leg", f"{fmt_p(la)}/hold · {lw * 100:.0f}% win",
              f"{len(longs):,} positions")
    d2.metric("Short leg", f"{fmt_p(sa)}/hold · {sw * 100:.0f}% win",
              f"{len(shorts):,} positions")

# ---------------- equity vs SPY ----------------
st.subheader("Equity vs SPY")
dates = pd.to_datetime(res["dates"])
fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(x=dates, y=res["bench"]["curve"], name="SPY buy & hold",
                            line=dict(color="#64748b", width=1.5)))
fig_eq.add_trace(go.Scatter(x=dates, y=res["strat"]["curve"],
                            name="Sentiment strategy",
                            line=dict(color="#38bdf8", width=2)))
fig_eq.update_layout(yaxis_title="Growth of $1", hovermode="x unified",
                     legend=dict(orientation="h", y=1.02),
                     margin=dict(l=40, r=10, t=10, b=10), height=380)
st.plotly_chart(fig_eq, width="stretch")
st.caption(
    f"Strategy equity starts earning at the first event's entry; SPY is buy-and-hold "
    f"over the same window. Costs of {cost_bps} bps per entry and exit leg are charged "
    "on trade days and are already reflected in the blue line."
)

# ---------------- PEAD drift chart ----------------
st.subheader("Event study — the PEAD plot")
st.caption("Average cumulative return by tone quintile, days after entry. "
           "If tone predicts drift, the teal (most positive) line should sit above "
           "the red (most negative) line and the gap should widen over the holding "
           "window. Quintiles are ranked on the full sample for display; the traded "
           "portfolio never sees future scores.")
drift_cols = ["#f87171", "#fb923c", "#94a3b8", "#34d399", "#22d3ee"]
fig_d = go.Figure()
for dq in res["drift"]:
    q = dq["q"]
    label = (f"Q{q + 1} {'(most negative)' if q == 0 else '(most positive)' if q == 4 else ''} "
             f"· n={dq['n']} · score {dq['lo']:.2f}…{dq['hi']:.2f}")
    xs = [d for d, v in enumerate(dq["cum"]) if v is not None]
    ys = [dq["cum"][d] for d in xs]
    fig_d.add_trace(go.Scatter(x=xs, y=ys, name=label,
                               line=dict(color=drift_cols[q],
                                         width=2.5 if q in (0, 4) else 1.5)))
fig_d.add_hline(y=0, line_dash="dash", line_color="#64748b", line_width=1)
fig_d.update_layout(xaxis_title="Days after entry", yaxis_title="Avg cum return",
                    hovermode="x unified", legend=dict(orientation="h", y=1.02),
                    margin=dict(l=40, r=10, t=10, b=10), height=380)
fig_d.update_yaxes(tickformat=".0%")
st.plotly_chart(fig_d, width="stretch")

# ---------------- monthly PnL + yearly table ----------------
st.subheader("Monthly PnL")
m = pd.DataFrame(res["monthly"])
colors = ["#34d399" if r >= 0 else "#f87171" for r in m["r"]]
fig_m = go.Figure(go.Bar(x=m["m"], y=m["r"], marker_color=colors))
fig_m.update_layout(yaxis_title="Return", hovermode="x",
                    margin=dict(l=40, r=10, t=10, b=60), height=260)
fig_m.update_yaxes(tickformat=".0%")
fig_m.update_xaxes(tickangle=45, tickmode="linear")
st.plotly_chart(fig_m, width="stretch")

st.subheader("Per-year stats")
ydf = pd.DataFrame(res["yearly"])
ydf = ydf.rename(columns={"y": "Year", "r": "Return", "sharpe": "Sharpe",
                          "maxdd": "MaxDD", "n": "Days"})
ydf["Return"] = ydf["Return"].map(lambda x: fmt_p(x))
ydf["MaxDD"] = ydf["MaxDD"].map(lambda x: fmt_p(x))
ydf["Sharpe"] = ydf["Sharpe"].map(lambda x: f"{x:.2f}")
st.dataframe(ydf, width="stretch", hide_index=True)

# ---------------- extreme events ----------------
st.subheader("Extreme-tone events")
st.caption("10 most positive / 10 most negative scores, all time.")


def _ev_df(rows, show_side):
    d = pd.DataFrame(rows)
    out = pd.DataFrame({
        "Ticker": d["t"],
        "Sector": d["sec"],
        "Filed": d["fd"],
        "Entry": d["ed"],
        "Score": d["s"].map(lambda x: f"{x:.3f}"),
        f"{hold}-day fwd ret": d["fwd"].map(lambda x: fmt_p(x)),
    })
    if show_side:
        out["Side"] = d["side"].map(lambda x: "LONG" if x > 0 else "SHORT" if x < 0 else "—")
        out = out[["Ticker", "Sector", "Filed", "Entry", "Score", "Side",
                   f"{hold}-day fwd ret"]]
    return out


st.dataframe(_ev_df(res["topS"], show_side=False), width="stretch",
             hide_index=True)
st.markdown("— most negative —")
st.dataframe(_ev_df(res["botS"], show_side=False), width="stretch",
             hide_index=True)

with st.expander("Recent events (last 40) with forward returns"):
    st.dataframe(_ev_df(res["recent"], show_side=True), width="stretch",
                 hide_index=True)

# ---------------- how it works ----------------
with st.expander("The strategy in plain language", expanded=True):
    st.markdown("""
- **The idea (PEAD):** after earnings news, stock prices drift in the direction of the surprise for weeks — the market digests news slowly. This lab bets the *tone* of the press release predicts the drift: cheerful releases drift up, gloomy ones drift down.
- **Scoring:** each 8-K item 2.02 press release is counted against the Loughran-McDonald finance word list (positive vs negative words). Score = (pos − neg) / (pos + neg + 1).
- **Ranking without peeking:** at each event's entry date, its score is ranked against the trailing 12 months of scores — only past releases. Top tail → long candidate, bottom tail → short candidate. Entry is the *next trading day's close after the SEC filing*, never the announcement day.
- **Portfolio:** equal-weight among active candidates (max N per side), each held exactly H trading days. Long/short mode targets 50¢ long / 50¢ short per dollar (dollar-neutral); long-only mode is 100% long.
- **Costs:** 10 bps haircut (adjustable) on every entry and exit leg, charged on the trade days.
""")

with st.expander("Honest limitations (read before trusting it)"):
    st.markdown("""
- **PEAD has decayed.** The anomaly was strongest in the 1980s–2000s; published research and practitioner notes show it has shrunk as markets got faster and more algorithmic. A lexicon is cruder than the earnings-surprise measures in the literature.
- **Lexicon ≠ understanding.** Word counting misses negation, sarcasm, and context ("loss" in "loss prevention program"). It is a tone proxy, not comprehension.
- **Filing-date timing.** We enter at the next trading day's close after the EDGAR filing timestamp. If news leaked or the release hit the wires before filing, part of the move is already gone — our timing is conservative but not exact.
- **Small quarterly samples.** ~90 names × 4 quarters means each ranking rests on ~90 events; terciles are ~30 names each. Thin samples = noisy results.
- **Shorting is expensive in real life.** The backtest charges bps only — real shorting adds borrow fees, hard-to-borrow risk, and dividends owed.
- **Survivorship bias:** the universe is today's S&P 500 list; delisted names are missing, which flatters results.
- **Universe gaps:** banks (BAC, C, GS, JPM, MS, WFC) and NVDA were dropped from the price panel (insufficient 8-K item 2.02 text events or missing prices), so the universe skews away from financials and the biggest AI name.
- **Simplifications:** trades at the daily close, no slippage beyond the bps haircut, no corporate-action edge cases.
- Backtested performance does not predict future results. Educational research tool, **not investment advice**.
""")
