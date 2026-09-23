# Quant Strategy Lab

A unified Streamlit dashboard consolidating five quantitative trading research
tools (ported from the original single-file HTML labs).

## Modules

1. **Pairs Trading** — cointegration stat-arb scanner + z-score backtests
2. **ML Signals** — walk-forward logistic-regression return prediction
3. **RL Agent** — Q-learning trading agent on SPY
4. **Sentiment Signals** — lexicon scorer + market-regime classifier
5. **Sentiment Strategy** — post-earnings drift strategy on 2,515 SEC filings

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or with the existing project venv:

```bash
~/workspace/venvs/pfapp/bin/pip install -r requirements.txt
~/workspace/venvs/pfapp/bin/streamlit run app.py
```

## Notes

- All data is embedded in `quantlab/data/` (~8 MB); no API keys required.
- Every backtest models 10 bps transaction costs and is strictly point-in-time
  (no look-ahead).
- These are **research tools, not investment advice**. Several strategies
  underperform the benchmark — the apps show the honest numbers, including
  survivorship-bias and cost caveats on each page.
- The Sentiment Signals page has optional, clearly-labeled live-data expanders
  (yfinance VIX/SPY, SEC 8-K lookup, LLM scoring with your own API key); the
  app works fully offline without them.
