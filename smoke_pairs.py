"""Real-data smoke test — Python mirror of quant-lab-build/pairs/sanity_real.js."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantlab import pairs as E

DATA = json.load(open("quantlab/data/daily.json"))
print("DATA:", len(DATA["px"]), "tickers x", len(DATA["dates"]), "days",
      DATA["dates"][0], "->", DATA["dates"][-1])

t0 = time.time()
train = E.slice_data(DATA, "2018-01-01", "2021-12-31")
test = E.slice_data(DATA, "2022-01-01", "2026-08-31")
print("train days:", len(train["dates"]), " test days:", len(test["dates"]))

rows = E.scan_pairs(train)
print("candidates:", len(rows), "(%.2fs)" % (time.time() - t0))

P = {"entryZ": 2, "exitZ": 0.25, "stopZ": 4, "maxHoldMult": 2, "costBps": 10}
for r in rows:
    bt = E.backtest_pair(train, r["a"], r["b"], r["beta"], {**P, "hl": r["hl"]})
    r["isSharpe"] = bt["stats"]["sharpe"]
ranked = sorted(rows, key=lambda r: -r["isSharpe"])
print("top 8 by IS Sharpe:")
for r in ranked[:8]:
    print(f"  {r['a']}/{r['b']} {r['sector']} adf={r['adf']} hl={r['hl']} "
          f"isSharpe={r['isSharpe']:.2f}")

top5 = ranked[:5]
pf = E.backtest_portfolio(test, top5, 5, P)
print("\nOUT-OF-SAMPLE top-5 portfolio (2022-26):")
print("  strategy:", pf["stats"])
print("  SPY     :", pf["spyStats"])
print("  trades  :", len(pf["trades"]))

exp = {"sharpe": 0.35, "cagr": 0.0275, "maxdd": -0.1129, "win": 0.583}
got = pf["stats"]
checks = [
    ("Sharpe", abs(got["sharpe"] - exp["sharpe"]) <= 0.1),
    ("CAGR", abs(got["cagr"] - exp["cagr"]) / abs(exp["cagr"]) <= 0.15),
    ("maxDD", abs(got["maxdd"] - exp["maxdd"]) / abs(exp["maxdd"]) <= 0.15),
    ("win", abs(got["win"] - exp["win"]) / abs(exp["win"]) <= 0.15),
]
print("\nsanity checks vs HTML reference:")
ok_all = True
for name, ok_ in checks:
    print(f"  {name}: {'PASS' if ok_ else 'FAIL'}")
    ok_all = ok_all and ok_
sys.exit(0 if ok_all else 1)
