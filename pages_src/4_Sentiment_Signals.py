"""Quant Strategy Lab — Module 4: Sentiment Signals & Regime Classifier.

LLM-*style* sentiment layer (lexicon scorer, offline) + a point-in-time
VIX/DMA regime classifier with a strategy-switch map.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quantlab import data as qd
from quantlab import sentiment as qs

st.set_page_config(page_title="Module 4: Sentiment Signals", page_icon="🧪", layout="wide")

st.title("🧪 Module 4: Sentiment Signals & Regime Classifier")
st.markdown(
    "An offline-first, LLM-*style* sentiment layer plus a market-regime "
    "classifier with a strategy-switch map. The lexicon scorer, the sample "
    "events, and the regime engine run locally on embedded data — they never "
    "touch the network."
)
st.warning(
    "⚠️ **Honest framing:** sentiment here is a **confirmation layer**, not "
    "standalone alpha. The evidence that retail investors can earn "
    "stock-picking alpha from LLM/LLM-style sentiment scores is **weak** — "
    "press releases are public at filing time, so scoring them confirms an "
    "event, it doesn't predict the market. Nothing here is investment advice."
)

lexicon = qd.load_lexicon()
samples = qd.load_samples()
sdata = qd.load_sentiment_data()
regimes = qs.classify_regimes(sdata["dates"], sdata["spy"], sdata["vix"])

CAT_LABELS = [("pos", "Positive", "#34d399"), ("neg", "Negative", "#f87171"),
              ("unc", "Uncertain", "#fbbf24"), ("lit", "Litigious", "#38bdf8")]

# ---------------------------------------------------------------- (a) scorer
st.header("(a) Text sentiment scorer — 100% offline")
st.caption(
    "Paste an earnings press release or news article. A compact financial word "
    "list (~166 words, in the spirit of Loughran–McDonald) counts positive / "
    "negative / uncertain / litigious language and scores it. Crude but "
    "transparent — you see exactly which words matched."
)

sample_names = [f"SAMPLE — {s['company']} · {s['quarter']} · {s['title']}" for s in samples]
c_pick, c_load = st.columns([3, 1])
with c_pick:
    sample_choice = st.selectbox("Sample earnings events (fictional, for demo)", sample_names)
with c_load:
    st.write("")
    st.write("")
    if st.button("Load sample", type="primary"):
        st.session_state["sent_text"] = samples[sample_names.index(sample_choice)]["text"]
        st.session_state.pop("sent_result", None)
        st.rerun()

if "sent_text" not in st.session_state:
    st.session_state["sent_text"] = ""
text = st.text_area("Earnings text", key="sent_text", height=170,
                    placeholder="Paste earnings press release or news text here…")

if st.button("Score text", type="primary"):
    st.session_state["sent_result"] = qs.score_text(text, lexicon) if text.strip() else None

res = st.session_state.get("sent_result")
if res is None:
    st.info("Paste some text (or load a sample) and press **Score text**.")
else:
    m1, m2, m3 = st.columns(3)
    net, words = res["net"], res["words"]
    tone = "Bullish tone" if net >= 0.2 else "Bearish tone" if net <= -0.2 else "Neutral / mixed tone"
    tone_color = "#34d399" if net >= 0.2 else "#f87171" if net <= -0.2 else "#fbbf24"
    m1.metric("Net sentiment (pos−neg)/(pos+neg+1)", f"{net:+.3f}")
    m1.markdown(f"<span style='color:{tone_color};font-weight:700'>{tone}</span>", unsafe_allow_html=True)
    m2.metric("Words tokenized", words)
    hits = sum(res["counts"].values())
    m3.metric("Lexicon hits", hits)

    fig = go.Figure(go.Bar(
        x=[res["per1000"][c] for c, _, _ in CAT_LABELS],
        y=[label for _, label, _ in CAT_LABELS],
        orientation="h",
        marker_color=[col for _, _, col in CAT_LABELS],
        text=[f"{res['per1000'][c]:.1f} /1k" for c, _, _ in CAT_LABELS],
        textposition="outside",
    ))
    fig.update_layout(title="Hits per 1,000 words", height=260,
                      margin=dict(l=90, r=80, t=40, b=20), xaxis_title="per 1,000 words")
    st.plotly_chart(fig, width="stretch")

    cols = st.columns(4)
    for (c, label, color), col in zip(CAT_LABELS, cols):
        with col:
            with st.expander(f"{label} ({res['counts'][c]})", expanded=(c in ("pos", "neg"))):
                matched = res["matched"][c]
                if not matched:
                    st.caption("no matches")
                else:
                    st.markdown(" ".join(
                        f"`{w['word']}` ×{w['count']}" for w in matched))
    st.caption("Tokenized: lowercased, punctuation stripped. Scores normalized per 1,000 words. "
               "No negation handling — “not weak” still counts as a negative word.")

# ---------------------------------------------------------------- (b) regimes
st.header("(d) Regime classifier + strategy-switch map — offline")
st.caption(
    "Point-in-time daily regime from SPY and VIX, using only data known up to "
    "each day. **Crisis:** VIX > 35, or SPY below its 200-day average with VIX "
    "> 25. **Bear:** SPY below 200-day average. **Bull:** SPY above 50-day "
    "average and 50-day above 200-day. Otherwise **sideways**. "
    "(First 199 days: n/a — not enough history for a 200-day average.)"
)

last = regimes[-1]
REG_BADGE = {"bull": ("#34d399", "success"), "bear": ("#f87171", "error"),
             "sideways": ("#fbbf24", "warning"), "crisis": ("#ff2d2d", "error"),
             "n/a": ("#94a3b8", "info")}
color, box = REG_BADGE[last["regime"]]
callout = (f"**Current regime ({last['date']}):** "
           f"<span style='color:{color};font-weight:800'>{last['regime'].upper()}</span> "
           f"· VIX {last['vix']} · SPY 50DMA {last['dma50']} · 200DMA {last['dma200']}")
getattr(st, box)(callout)

st.subheader("Strategy-switch map")
st.table({
    "Regime": ["🟢 BULL", "🔴 BEAR", "🟡 SIDEWAYS", "🚨 CRISIS"],
    "Playbook": [
        "Momentum / breakouts — ride trends, buy strength, earnings-breakout setups",
        "Defensive quality — cash-flow compounders, low leverage, dividends",
        "Mean reversion / pairs — buy the range bottom, sell the top",
        "Cash / minimal / hedge — capital preservation first",
    ],
    "Sizing / risk": [
        "Full size; standard stops",
        "Smaller size; wider stops (volatility is higher)",
        "Standard size; tight exits at range edges",
        "Minimal exposure; hedge or stand aside",
    ],
    "Nearest lab module": [
        "2️⃣ ML Signals (momentum predictions) · 5️⃣ Sentiment Strategy (long leg)",
        "— no module implements defensive quality (3️⃣ RL Agent is the lowest-vol option)",
        "1️⃣ Pairs Trading (mean reversion / stat arb)",
        "— stand aside; no module",
    ],
})

counts = qs.regime_counts(regimes)
classified = sum(counts[r] for r in ("bull", "bear", "sideways", "crisis"))
sc1, sc2, sc3, sc4 = st.columns(4)
for col, r, lab, colr in [(sc1, "bull", "Bull days", "#34d399"),
                          (sc2, "bear", "Bear days", "#f87171"),
                          (sc3, "sideways", "Sideways days", "#fbbf24"),
                          (sc4, "crisis", "Crisis days", "#ff2d2d")]:
    pct = counts[r] / classified * 100 if classified else 0
    col.metric(lab, f"{counts[r]:,}", f"{pct:.1f}% of classified days")

tickers = ["SPY"] + list(sdata["prices"].keys())
ticker = st.selectbox("Timeline ticker", tickers,
                      help="Price (adjusted closes) with the regime band underneath — see how often regimes flip.")
px = sdata["spy"] if ticker == "SPY" else sdata["prices"][ticker]

# contiguous regime segments for background bands
segments = []
start = 0
for i in range(1, len(regimes) + 1):
    if i == len(regimes) or regimes[i]["regime"] != regimes[start]["regime"]:
        segments.append((start, i - 1, regimes[start]["regime"]))
        start = i

RCOL = {"bull": "rgba(52,211,153,0.18)", "bear": "rgba(248,113,113,0.18)",
        "sideways": "rgba(251,191,36,0.18)", "crisis": "rgba(255,45,45,0.30)",
        "n/a": "rgba(148,163,184,0.10)"}
fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                     vertical_spacing=0.04,
                     subplot_titles=(f"{ticker} adjusted close — regime background", "VIX"))
for a, b, r in segments:
    fig2.add_vrect(x0=sdata["dates"][a], x1=sdata["dates"][b], fillcolor=RCOL[r],
                   line_width=0, row=1, col=1, layer="below")
fig2.add_trace(go.Scatter(x=sdata["dates"], y=px, name=ticker,
                          line=dict(color="#38bdf8", width=1.4)), row=1, col=1)
fig2.add_trace(go.Scatter(x=sdata["dates"],
                          y=[r["dma50"] for r in regimes], name="50DMA",
                          line=dict(color="#fbbf24", width=1, dash="dash")), row=1, col=1)
fig2.add_trace(go.Scatter(x=sdata["dates"],
                          y=[r["dma200"] for r in regimes], name="200DMA",
                          line=dict(color="#f87171", width=1, dash="dash")), row=1, col=1)
fig2.add_trace(go.Scatter(x=sdata["dates"], y=sdata["vix"], name="VIX",
                          line=dict(color="#c084fc", width=1.2)), row=2, col=1)
for lvl, lab in [(35, "crisis line"), (25, "stress line")]:
    fig2.add_hline(y=lvl, line=dict(color="#64748b", dash="dot", width=1),
                   annotation_text=lab, row=2, col=1)
fig2.update_layout(height=560, margin=dict(t=50, b=20),
                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                   hovermode="x unified")
fig2.update_xaxes(rangeslider_visible=False)
st.plotly_chart(fig2, width="stretch")
st.caption("🟩 bull · 🟥 bear · 🟨 sideways · 🟥 crisis · ⬜ n/a (warm-up). "
           "Regimes are descriptive, not predictive — a bull label today says nothing about tomorrow.")

# ---------------------------------------------------------------- (c) live data
st.header("Optional live data — offline-first")
st.caption("Everything above works without the network. These are clearly-labeled, "
           "best-effort extras.")

with st.expander("📡 Try live VIX/SPY via yfinance (optional)"):
    st.caption("Fetches 1 year of ^VIX + SPY from Yahoo and recomputes regimes. "
               "Falls back to the embedded data on any failure.")
    if st.button("Fetch live data"):
        try:
            import yfinance as yf
            lv = yf.download("^VIX", period="1y", progress=False, auto_adjust=False)["Close"]["^VIX"]
            ls = yf.download("SPY", period="1y", progress=False, auto_adjust=False)["Close"]["SPY"]
            dates = [d.strftime("%Y-%m-%d") for d in ls.index]
            vix_vals = [None if v != v else float(v) for v in lv.reindex(ls.index).tolist()]
            spy_vals = [float(v) for v in ls.tolist()]
            live_reg = qs.classify_regimes(dates, spy_vals, vix_vals)
            cur = live_reg[-1]
            st.success(f"Live regime ({cur['date']}): **{cur['regime'].upper()}** "
                       f"· VIX {cur['vix']} · SPY 50DMA {cur['dma50']} · 200DMA {cur['dma200']}")
            st.caption("Note: 1 year of history means the first 199 days are n/a; "
                       "only the most recent days are fully classified. Compare with the "
                       "embedded callout above — they use the same engine.")
        except ImportError:
            st.warning("yfinance is not installed (`pip install yfinance`) — "
                       "using the embedded data above instead.")
        except Exception as e:  # noqa: BLE001 - graceful offline fallback
            st.warning(f"Live fetch failed ({type(e).__name__}: {e}) — "
                       "using the embedded data above instead.")

with st.expander("📄 SEC 8-K lookup (optional, best-effort)"):
    st.caption("Attempts to list a company's recent 8-K filings via the SEC EDGAR "
               "submissions API. **Heads-up from the original browser lab:** SEC's "
               "API requires a User-Agent header that browsers can't set, and "
               "`data.sec.gov` generally doesn't allow cross-origin page requests — "
               "so in the browser version, failure was the expected outcome. "
               "Kept lightweight here: no fetch is attempted; if you need the "
               "filings, open EDGAR directly.")
    c1, c2 = st.columns(2)
    c1.text_input("Ticker", value="AAPL")
    c2.text_input("CIK", value="320193", help="Known CIKs: AAPL 320193 · MSFT 789019 · NVDA 1045810")
    st.caption("Browse: https://www.sec.gov/cgi-bin/browse-edgar — filter by 8-K.")

with st.expander("🤖 LLM scoring with your API key (optional)"):
    st.caption("A second-opinion score from an LLM on the text you scored in (a). "
               "**Deliberately not wired up:** your key would be kept in session "
               "memory only (never written to disk), calls cost money per your "
               "provider's pricing, and many providers (including OpenAI) block "
               "direct browser calls via CORS — you'd need a CORS-enabled endpoint "
               "or a small local proxy. LLM scores are not verified financial "
               "advice: the model can misread numbers, hallucinate, or overweight "
               "dramatic language. The offline lexicon score above is the "
               "reproducible baseline.")
    st.text_input("API key (session memory only, never stored)", type="password",
                  placeholder="sk-…", disabled=True)
    c3, c4 = st.columns(2)
    c3.text_input("Endpoint URL", value="https://api.openai.com/v1/chat/completions", disabled=True)
    c4.text_input("Model", value="gpt-4o-mini", disabled=True)
    st.button("Score with LLM", disabled=True,
              help="Disabled by design — see explanation above.")

# ---------------------------------------------------------------- caveats
st.header("How it works & limits — read before trusting any score")
st.markdown("""
- **Lexicon counting is crude.** There is **no negation handling** — "not weak"
  and "no losses" still count as negative words. It also misses sarcasm,
  context, and industry jargon. Treat it as a rough tone meter, not a judge.
- **Press releases are public at filing time.** Scoring an earnings release is
  **event confirmation, not prediction** — the price usually moves in the first
  minutes after the filing. Any "edge" would come from faster or better
  interpretation, not from the score itself.
- **LLM scores are not verified financial advice.** The model can misread
  numbers, hallucinate, or overweight dramatic language. Always read the
  rationale and the source text yourself.
- **The 8-K fetch is best-effort.** SEC's API needs a User-Agent header browsers
  can't set, and `data.sec.gov` generally doesn't allow cross-origin page
  requests — failure is the expected outcome in a browser.
- **Regimes are descriptive, not predictive.** The classifier labels the past
  cleanly (with point-in-time math, so no look-ahead), but a bull label today
  says nothing about tomorrow. The timeline shows how often regimes flip —
  whipsaws are real.
- **Backtest ≠ future.** Price data are split/dividend-adjusted closes from
  Yahoo, 2018-01-02 → 2026-08-31. This is research tooling,
  **not investment advice**.
""")
