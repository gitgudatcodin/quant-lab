"""Module 3 — RL Trading Agent.

Tabular Q-learning trader on SPY (2015→2026), ported from
quant-lab-build/rl (engine.js + template_ui.html). The agent plays one full
pass over history per episode, learning a 54-state x 3-action Q-table, then
the frozen greedy policy is replayed on unseen test data.
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from quantlab import rl
from quantlab import data as qd

SPLITS = {
    "a": {"label": "Train 2015–2021 · Test 2022–2026",
          "train_end": "2021-12-31", "test_end": "2026-08-31", "full": False},
    "b": {"label": "Train 2015–2023 · Test 2024–2026",
          "train_end": "2023-12-31", "test_end": "2026-08-31", "full": False},
    "c": {"label": "Full sample (train = test) — in-sample!",
          "train_end": "2026-08-31", "test_end": "2026-08-31", "full": True},
}

DEFAULTS = {"episodes": 60, "lam": 0.005, "cost_bps": 5,
            "alpha": 0.1, "gamma": 0.99, "split": "a", "seed": 30}


def _reward_fig(rewards):
    avg = []
    for i in range(len(rewards)):
        k0 = max(0, i - 9)
        avg.append(sum(rewards[k0:i + 1]) / (i + 1 - k0))
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=rewards, mode="lines", name="episode reward",
                             line=dict(color="#38bdf8", width=1.2)))
    fig.add_trace(go.Scatter(y=avg, mode="lines", name="trailing-10 average",
                             line=dict(color="#fbbf24", width=1.8)))
    fig.update_layout(height=300, margin=dict(l=50, r=15, t=10, b=40),
                      xaxis_title="episode", yaxis_title="total reward",
                      legend=dict(orientation="h", y=1.05))
    return fig


def _equity_fig(eq, eq_bh):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[d for d, _ in eq_bh], y=[v for _, v in eq_bh],
                             mode="lines", name="SPY buy & hold",
                             line=dict(color="#94a3b8", width=1.4)))
    fig.add_trace(go.Scatter(x=[d for d, _ in eq], y=[v for _, v in eq],
                             mode="lines", name="RL agent (greedy)",
                             line=dict(color="#34d399", width=1.8)))
    fig.update_layout(height=380, margin=dict(l=60, r=15, t=10, b=40),
                      yaxis_title="growth of $1",
                      legend=dict(orientation="h", y=1.05))
    return fig


def _action_fig(action_frac):
    fig = go.Figure(go.Bar(
        x=[f * 100 for f in action_frac], y=["SHORT", "FLAT", "LONG"],
        orientation="h", marker_color=["#f87171", "#94a3b8", "#34d399"],
        text=[f"{f * 100:.1f}%" for f in action_frac], textposition="outside"))
    fig.update_layout(height=220, margin=dict(l=70, r=40, t=10, b=40),
                      xaxis_title="% of test days")
    return fig


@st.cache_data(show_spinner="Training the RL agent…")
def _train_cached(alpha, gamma, lam, cost_bps, max_ep, split_key, seed,
                  _progress_cb=None, _progress_every=5):
    """Full train + evaluate pipeline, cached on all hyperparams + seed.

    _progress_cb / _progress_every are underscore-prefixed so they are
    excluded from the cache key; the callback only fires on a cache miss
    (live progress), while a cache hit returns instantly.
    """
    d = qd.load_spy_daily()
    dates, px = d["dates"], d["px"]
    F = rl.features(px)
    P = {"alpha": alpha, "gamma": gamma, "lambda": lam, "cost_bps": cost_bps}
    sp = SPLITS[split_key]
    i1 = rl.date_idx(dates, sp["train_end"])
    i0 = F["warm"]
    disc = rl.Discretizer(F, i0, i1)
    agent, rewards, n_ep, stopped = rl.train_agent(
        F, disc, i0, i1, P, max_ep, seed,
        progress_cb=_progress_cb, progress_every=_progress_every)
    if sp["full"]:
        j0, j1 = i0, i1
    else:
        j0, j1 = i1 + 1, rl.date_idx(dates, sp["test_end"])
    ev = rl.evaluate(dates, F, disc, agent, j0, j1, P)
    return {
        "rewards": rewards, "episodes": n_ep, "stopped_early": stopped,
        "Q": agent["Q"], "visits": agent["visits"],
        "m_lo": disc.m_lo, "m_hi": disc.m_hi, "v_med": disc.v_med,
        "i0": i0, "i1": i1, "j0": j0, "j1": j1,
        "eq": ev["eq"], "eq_bh": ev["eq_bh"], "trades": ev["trades"],
        "action_frac": list(ev["action_frac"]),
        "stats": ev["stats"], "stats_bh": ev["stats_bh"],
        "params": {"alpha": alpha, "gamma": gamma, "lambda": lam,
                   "cost_bps": cost_bps, "split": split_key, "seed": seed},
    }


# ---------------- sidebar ----------------
st.sidebar.header("Training")
episodes = st.sidebar.slider("Episodes", 10, 200, DEFAULTS["episodes"], 5)
lam = st.sidebar.slider("λ drawdown penalty", 0.0, 0.1, DEFAULTS["lam"], 0.005,
                        format="%.3f")
st.sidebar.caption(
    "λ is scaled to daily-return units: 0.005 ≈ one day's typical return at a "
    "10% drawdown. Crank it up and the penalty drowns out the learning signal "
    "and the policy degrades.")
cost_bps = st.sidebar.slider("Cost per position change (bps)", 0, 50,
                             DEFAULTS["cost_bps"], 1)
alpha = st.sidebar.slider("α learning rate", 0.01, 0.5, DEFAULTS["alpha"], 0.01)
gamma = st.sidebar.slider("γ discount", 0.90, 0.999, DEFAULTS["gamma"], 0.001)
st.sidebar.caption("HTML defaults: α = 0.1, γ = 0.99. ε decays 1.0 → 0.05. "
                   "Early stop if the trailing-10-episode average reward "
                   "stalls for 15 episodes.")

st.sidebar.header("Train / test split")
split_key = st.sidebar.selectbox(
    "Split", options=list(SPLITS.keys()),
    format_func=lambda k: SPLITS[k]["label"],
    index=list(SPLITS.keys()).index(DEFAULTS["split"]))
if SPLITS[split_key]["full"]:
    st.sidebar.warning("⚠ In-sample: the agent is tested on the data it "
                       "trained on. This shows what it *memorized*, not what "
                       "it can predict. Read it skeptically.")
seed = st.sidebar.number_input("Seed", min_value=0, max_value=999999,
                               value=DEFAULTS["seed"], step=1,
                               help="RL is stochastic (ε-greedy exploration): "
                                    "a fixed seed makes a run reproducible; "
                                    "different seeds give different results.")

# ---------------- main ----------------
st.title("🤖 RL Trading Agent")
st.markdown(
    "A reinforcement-learning trader on SPY (2015→2026). The agent plays a full "
    "pass over history as one **episode**: each day it reads the market's "
    "**state** (momentum, RSI, volatility regime, its own position), picks "
    "long / flat / short, and gets a **reward** = next-day P&L − trading costs "
    "− a penalty for drawdowns. Over many episodes it fills in a **Q-table** — "
    "its memory of what worked in each market situation. Then we test the "
    "learned policy on data it never trained on.")

res = st.session_state.get("rl3_res")
if st.sidebar.button("▶ Train agent", type="primary"):
    prog = st.progress(0.0, text="Training…")
    chart_ph = st.empty()

    def _cb(done, total, rewards):
        prog.progress(done / total, text=f"Training… episode {done}/{total}")
        chart_ph.plotly_chart(_reward_fig(rewards), width="stretch")

    res = _train_cached(alpha, gamma, lam, cost_bps, episodes, split_key, seed,
                        _progress_cb=_cb, _progress_every=5)
    prog.progress(1.0, text="Training complete")
    st.session_state["rl3_res"] = res

if res is not None:
    p = res["params"]
    st.subheader("Training: reward per episode")
    st.plotly_chart(_reward_fig(res["rewards"]), width="stretch")
    k0 = max(0, len(res["rewards"]) - 10)
    final_avg = sum(res["rewards"][k0:]) / (len(res["rewards"]) - k0)
    st.caption(
        f"Done: **{res['episodes']}** episodes"
        + (" — **early stop** (trailing-10 avg reward stalled 15 episodes)"
           if res["stopped_early"] else "")
        + f". Final trailing-10 avg reward: **{final_avg:.2f}**. "
        f"α={p['alpha']}, γ={p['gamma']}, λ={p['lambda']}, "
        f"cost={p['cost_bps']} bps, seed={p['seed']}.")

    st.subheader("Test-period results")
    s, b = res["stats"], res["stats_bh"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Agent CAGR", f"{s['cagr'] * 100:+.1f}%")
    c2.metric("Agent Sharpe", f"{s['sharpe']:.2f}")
    c3.metric("Agent max DD", f"{s['maxDD'] * 100:.1f}%")
    c4.metric("Trades", f"{res['trades']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Buy-hold CAGR", f"{b['cagr'] * 100:+.1f}%")
    c2.metric("Buy-hold Sharpe", f"{b['sharpe']:.2f}")
    c3.metric("Buy-hold max DD", f"{b['maxDD'] * 100:.1f}%")
    c4.metric("Episodes trained", f"{res['episodes']}")

    st.subheader("Equity: agent vs buy & hold — growth of $1 on the test period")
    st.plotly_chart(_equity_fig(res["eq"], res["eq_bh"]),
                    width="stretch")
    n_days = res["j1"] - res["j0"] + 1
    st.caption(
        f"Test window {res['eq'][0][0]} → {res['eq'][-1][0]} ({n_days} trading "
        f"days). Greedy policy, {p['cost_bps']} bps per position-change unit, "
        f"λ={p['lambda']}.")

    st.subheader("Action distribution — fraction of test days long / flat / short")
    st.plotly_chart(_action_fig(res["action_frac"]), width="stretch")

    st.subheader("Q-table peek — most-visited market states")
    rows = rl.q_peek({"Q": res["Q"], "visits": res["visits"]}, None, 8)
    df = pd.DataFrame([{
        "State": f"s{r['state']}", "Visits": r["visits"],
        "Momentum": r["mom"], "RSI": r["rsi"], "Vol regime": r["vol"],
        "Held pos": r["pos"], "Learned action": r["action"],
        "Q(short)": round(r["q"][0], 3), "Q(flat)": round(r["q"][1], 3),
        "Q(long)": round(r["q"][2], 3)} for r in rows])
    st.dataframe(df, width="stretch", hide_index=True)
    st.caption("State = the market situation the agent saw. “Learned action” is "
               "simply the highest Q-value in that row. Most states are visited "
               "rarely — the table is sparse, which is part of why this is a demo. "
               f"Discretizer cutoffs (train period): mom5 terciles "
               f"[{res['m_lo']:.4f}, {res['m_hi']:.4f}], vol-regime median "
               f"{res['v_med']:.3f}.")
else:
    st.info("Set the hyperparameters in the sidebar, then press **▶ Train agent**. "
            "Training takes a few seconds.")

with st.expander("How it works (plain language)"):
    st.markdown(
        "- The agent lives one trading day at a time. At each close it looks at 4 "
        "things: **momentum** (are the last 5 days up or down — weak/mid/strong by "
        "training-period terciles), **RSI** (oversold <35 / neutral / overbought >65), "
        "**volatility regime** (is the last 20 days' volatility high or low vs the last "
        "60 days), and **its own position** (short/flat/long). That's 3×3×2×3 = "
        "**54 possible situations**.\n"
        "- For every situation it keeps 3 numbers — the Q-values for going short, "
        "flat, or long. These start at zero and are updated by trial and error: take "
        "an action, see the reward, nudge the number toward `reward + γ × best future "
        "value`.\n"
        "- The **reward** shapes behavior: next-day profit on the position, minus a "
        "trading-cost haircut (bps per unit of position change — flipping long→short "
        "costs double), minus **λ × current drawdown**, which teaches caution: deep "
        "holes hurt even if the agent later recovers.\n"
        "- Early on it explores randomly (ε = 1.0); by the end it mostly exploits "
        "what it learned (ε = 0.05). One **episode** = one full pass over the "
        "training years. Then the frozen, greedy policy (always pick the best Q) is "
        "replayed on the unseen test years.")
with st.expander("⚠ Honest limits — read before trusting any number above"):
    st.markdown(
        "- **Runs vary.** ε-greedy exploration makes every training run different — "
        "change the seed and the test numbers move, sometimes a lot. One good run "
        "can be luck.\n"
        "- **RL memorizes regimes.** 2015–2021 was mostly a rising market with a few "
        "shocks. The agent “learns” what worked *then*; if the next regime differs, "
        "the memorized table can fail hard.\n"
        "- **54 states is tiny.** Real markets aren't Markov in 4 features. This is a "
        "teaching demo of Q-learning mechanics, not a strategy.\n"
        "- **Test results are one sample.** A good test-period number can be luck — "
        "especially the 2015–2023 / 2024–2026 split, which is only ~2.5 years of test "
        "data.\n"
        "- **Train ≠ test.** In-sample (train = test) numbers show memorization, not "
        "prediction.\n"
        "- **Costs are modeled, not real.** A flat bps haircut ignores spreads, "
        "slippage, borrow fees on shorts, and market impact. Real shorting of SPY "
        "costs more than this.\n"
        "- Lookahead discipline is enforced (all features use data known at the close; "
        "tercile cutoffs come from the train period only), but that doesn't make the "
        "future look like the past.\n"
        "- Backtested performance does not predict future results. This is research, "
        "not investment advice.")
