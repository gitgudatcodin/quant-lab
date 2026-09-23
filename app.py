"""Quant Strategy Lab — home page."""
import streamlit as st

st.set_page_config(
    page_title="Quant Strategy Lab",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Quant Strategy Lab")
st.markdown(
    """
A research workbench of five quantitative trading strategies, ported from the
original single-file HTML labs into one Streamlit app. Every strategy is
backtested **in-sample honestly**: transaction costs modeled, strictly
point-in-time (no look-ahead), benchmarked against SPY buy-and-hold.
"""
)

st.warning(
    "**Research tools, not investment advice.** Backtests are not predictions. "
    "Several of these strategies lose money or underperform the benchmark — "
    "that is the honest finding, and it is shown, not hidden. Costs are "
    "modeled at 10 bps but short borrow fees, dividends, and market impact "
    "are not. Survivorship bias applies wherever today's index lists are used."
)

cards = [
    (
        "1️⃣ Pairs Trading",
        "ML-flavored stat arb: same-sector cointegration scan (Engle-Granger ADF), "
        "z-score entry ±2 / exit 0.25 / stop ±4, dollar-neutral, 10 bps/leg.",
        "Top-5 pairs, out-of-sample 2022→2026: **Sharpe 0.35, CAGR +2.75%, "
        "maxDD −11.3%, win 58%** vs SPY 0.84 / +13.9%. A humbling baseline — "
        "pairs trading is hard.",
    ),
    (
        "2️⃣ ML Signals",
        "L2 logistic regression, walk-forward trained (36-mo train → 12-mo test, "
        "2020→2026) on 20 price + fundamental features. Top-decile predicted "
        "portfolio, monthly rebalance.",
        "**CAGR 36.5%, Sharpe 1.08** vs SPY 15.4% / 0.94 — but flattered by "
        "survivorship bias (today's S&P 500 list) and a momentum-regime window. "
        "Read skeptically.",
    ),
    (
        "3️⃣ RL Agent",
        "Tabular Q-learning (54 states: 5-day momentum tercile × RSI bucket × "
        "vol regime × position) trained on daily SPY 2015→2026, reward = "
        "position return − costs − λ·drawdown.",
        "Test 2022→2026: **CAGR +6.3%, Sharpe 0.83, maxDD −7.6%** vs buy-and-hold "
        "+12.2% / 0.75 / −24.5%. Lower return, much smoother ride. Runs vary — "
        "that's real stochasticity.",
    ),
    (
        "4️⃣ Sentiment Signals",
        "Loughran-McDonald-style lexicon scorer for earnings/news text plus a "
        "VIX + 50/200-DMA regime classifier with a strategy-switch map.",
        "Sample separation: beat-and-raise **+0.94** vs guide-down **−0.94**. "
        "Current regime: **BULL** (VIX 14.92 as of 2026-08-31). A confirmation "
        "layer, not standalone alpha.",
    ),
    (
        "5️⃣ Sentiment Strategy",
        "Post-earnings-announcement drift: 2,515 real SEC earnings releases "
        "(2019→2026, 90 names) scored by tone; long top-tercile, short "
        "bottom-tercile, 30-day holds, trailing-12-month ranking (no look-ahead).",
        "Default long/short **loses money** (CAGR −4.06%, Sharpe −0.44) because "
        "the short leg fails (−3.62%, 40% win). **Long-only: CAGR 17.39%, "
        "Sharpe 0.87 — beats SPY.**",
    ),
]

cols = st.columns(2)
for i, (title, desc, result) in enumerate(cards):
    with cols[i % 2]:
        st.subheader(title)
        st.markdown(desc)
        st.info(result)

st.divider()
st.markdown(
    """
### How to use
Pick a module from the sidebar. Each page has its own controls, charts, and
honest caveats. Data is embedded in `quantlab/data/` (no API keys needed);
the Sentiment Signals page has optional, clearly-labeled live-data expanders.

### Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```
"""
)
