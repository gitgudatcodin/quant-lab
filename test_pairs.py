"""Python port of quant-lab-build/pairs/test_engine.js — run with: python3 test_pairs.py"""
import math
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantlab import pairs as E

fails = 0


def ok(cond, name):
    global fails
    print(("PASS" if cond else "FAIL") + "  " + name)
    if not cond:
        fails += 1


# deterministic PRNG (same LCG as the JS test)
_seed = [42]


def rnd():
    _seed[0] = (_seed[0] * 1103515245 + 12345) & 0x7FFFFFFF
    return _seed[0] / 0x7FFFFFFF


def randn():
    return (rnd() + rnd() + rnd() + rnd() - 2) * 1.7


# --- 1. ols ---
x = [1, 2, 3, 4, 5]
y = [2 * v + 1 for v in x]
o = E.ols(x, y)
ok(abs(o["slope"] - 2) < 1e-9 and abs(o["intercept"] - 1) < 1e-9, "ols recovers y=2x+1")

# --- 2. ADF on known cointegrated pair vs independent random walks ---
N = 600
B = [100.0]
for i in range(1, N):
    B.append(B[-1] + randn())
e = 0.0
A = []  # A = 1.5*B + stationary AR(1) noise
for i in range(N):
    e = 0.9 * e + randn()
    A.append(1.5 * B[i] + e)
r1 = E.ols(B, A)
resid1 = [a - r1["slope"] * b for a, b in zip(A, B)]
adf_coint = E.adf(resid1)
ok(adf_coint < -2.9, f"adf passes on cointegrated pair (stat={adf_coint:.2f} < -2.9)")

C = [100.0]
D = [50.0]
for i in range(1, N):
    C.append(C[-1] + randn())
    D.append(D[-1] + randn())
r2 = E.ols(D, C)
resid2 = [c - r2["slope"] * d for c, d in zip(C, D)]
adf_rw = E.adf(resid2)
ok(adf_rw > -2.9, f"adf fails on independent random walks (stat={adf_rw:.2f} > -2.9)")

# --- 3. halflife: AR(1) with lambda=-0.1 -> hl = -ln2/ln(0.9) ~ 6.58 ---
h = 0.0
hs = []
for i in range(5000):
    h = 0.9 * h + randn()
    hs.append(h)
hl = E.halflife(hs)
ok(abs(hl - 6.58) < 1.0, f"halflife ~6.6 for AR(1) rho=0.9 (got {hl:.2f})")
ok(E.halflife(C) > 100, "halflife very large for random walk (got %.0f)" % E.halflife(C))

# --- 4. scanPairs on synthetic DATA (chunked scanner too) ---
dates = [(date(2020, 1, 1) + timedelta(days=i)).isoformat() for i in range(400)]
DATA = {
    "dates": dates,
    "px": {"AAA": A[:400], "BBB": B[:400], "CCC": C[:400], "DDD": D[:400]},
    "spy": B[:400],
    "sectors": {"AAA": "Tech", "BBB": "Tech", "CCC": "Tech", "DDD": "Tech"},
}
rows = E.scan_pairs(DATA)
found_ab = any((r["a"], r["b"]) in (("AAA", "BBB"), ("BBB", "AAA")) for r in rows)
found_cd = any((r["a"], r["b"]) in (("CCC", "DDD"), ("DDD", "CCC")) for r in rows)
ok(found_ab, "scan_pairs finds the cointegrated AAA/BBB pair")
ok(not found_cd, "scan_pairs rejects independent CCC/DDD walks")
print("   scanned rows:", rows)

sc = E.create_scanner(DATA)
steps = 0
while True:
    r = sc.step(1)
    steps += 1
    if r["done"]:
        break
ok(r["done"] and len(r["rows"]) == len(rows) and steps == r["total"],
   f"chunked scanner: {steps} steps, total={r['total']}, kept={r['kept']}")

# --- 5. backtestPair on synthetic mean-reverting pair ---
P = {"entryZ": 2, "exitZ": 0.25, "stopZ": 4, "maxHoldMult": 2, "costBps": 10}
bt = E.backtest_pair(DATA, "AAA", "BBB", rows[0]["beta"] if rows else 1.5, {**P, "hl": 10})
ok(len(bt["equity"]) == 340 and len(bt["trades"]) > 0,
   f"backtest_pair runs: {len(bt['equity'])} equity pts, {len(bt['trades'])} trades")
ok(all(t["entryD"] and t["exitD"] and math.isfinite(t["retPct"]) for t in bt["trades"]),
   "trade log fields sane")
ok(math.isfinite(bt["stats"]["sharpe"]) and math.isfinite(bt["stats"]["cagr"]), "stats finite")

# --- 6. backtestPortfolio ---
prs = [{"a": "AAA", "b": "BBB", "beta": 1.5, "hl": 10},
       {"a": "CCC", "b": "DDD", "beta": 1, "hl": 20}]
pf = E.backtest_portfolio(DATA, prs, 2, P)
ok(len(pf["equity"]) == len(pf["spy"]) and len(pf["equity"]) > 0,
   "portfolio equity aligns with SPY")
ok(len(pf["perPair"]) == 2 and pf["stats"]["n"] >= 0,
   "portfolio aggregates per-pair results")

# --- 7. sliceData ---
sl = E.slice_data(DATA, dates[100], dates[199])
ok(len(sl["dates"]) == 100 and sl["dates"][0] == dates[100]
   and len(sl["px"]["AAA"]) == 100 and len(sl["spy"]) == 100,
   "slice_data windows correctly")

# --- 8. stats ---
eq = [{"d": f"d{i}", "v": 1.1 ** (i / 251)} for i in range(252)]
st = E.stats(eq, [{"retPct": 1}, {"retPct": -1}, {"retPct": 2}])
ok(abs(st["cagr"] - 0.1) < 0.005 and abs(st["win"] - 2 / 3) < 0.01 and st["n"] == 3,
   f"stats: CAGR~10% win 2/3 (got {st['cagr']}, {st['win']})")

print("\nALL TESTS PASSED" if fails == 0 else f"\n{fails} TEST(S) FAILED")
sys.exit(1 if fails else 0)
