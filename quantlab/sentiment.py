"""Quant Strategy Lab — Module 4: sentiment engine.

Faithful pure-Python port of ``quant-lab-build/sentiment/engine.js``
(``tokenize`` / ``scoreText`` / ``classifyRegimes`` / ``regimeCounts``).

No streamlit, no network, no third-party imports — stdlib only.

Rounding note: the JS engine formats with ``Number(x.toFixed(d))`` (round
half away from zero).  ``_to_fixed`` replicates that exactly instead of
Python's banker's rounding, so parity holds even on .x5 boundaries.
"""

from __future__ import annotations

import math
import re

CATS = ("pos", "neg", "unc", "lit")


def _to_fixed(x: float, digits: int) -> float:
    """Replicate JS Number(x.toFixed(digits)): round half away from zero."""
    m = 10 ** digits
    sign = 1.0 if x >= 0 else -1.0
    return math.floor(abs(x) * m + 0.5) / m * sign


def tokenize(text: str) -> list[str]:
    """lowercase; [^a-z\\s-] -> space; split on whitespace; drop tokens <=1 char."""
    return [w for w in re.split(r"\s+", re.sub(r"[^a-z\s-]", " ", (text or "").lower())) if len(w) > 1]


def score_text(text: str, lexicon: dict) -> dict:
    """Score text with a word-count lexicon -> {words, counts, per1000, net, matched}.

    Each lexicon word matches whole tokens with hyphens stripped on both sides
    (so lexicon entry "write-down" matches token "writedown").  ``matched`` holds
    the top-12 words per category sorted by count desc then alphabetically.
    net = (pos - neg) / (pos + neg + 1), rounded to 3 decimals.
    """
    toks = tokenize(text)
    n = len(toks)
    stripped = [t.replace("-", "") for t in toks]

    counts = {c: 0 for c in CATS}
    word_counts: dict[str, int] = {}
    for c in CATS:
        for w in lexicon[c]:
            w_norm = w.replace("-", "")
            wc = sum(1 for t in stripped if t == w_norm)
            word_counts[w] = wc
            counts[c] += wc

    per1000 = {c: (_to_fixed(counts[c] / n * 1000, 1) if n else 0) for c in CATS}
    net = _to_fixed((counts["pos"] - counts["neg"]) / (counts["pos"] + counts["neg"] + 1), 3)

    matched: dict[str, list[dict]] = {}
    for c in CATS:
        items = sorted(
            ({"word": w, "count": word_counts[w]} for w in lexicon[c] if word_counts[w] > 0),
            key=lambda d: (-d["count"], d["word"]),
        )[:12]
        matched[c] = items

    return {"words": n, "counts": counts, "per1000": per1000, "net": net, "matched": matched}


def classify_regimes(dates: list, spy: list, vix: list) -> list[dict]:
    """Point-in-time daily regime from SPY closes and VIX -> [{date, regime, vix, dma50, dma200}].

    crisis: vix > 35 OR (spy < dma200 AND vix > 25)
    bear:   spy < dma200
    bull:   spy > dma50 AND dma50 > dma200
    else:   sideways
    Days with < 200 prior closes -> regime "n/a".  Moving averages use data up
    to day t only (strictly point-in-time, no look-ahead).
    """
    out: list[dict] = []
    sum50 = 0.0
    sum200 = 0.0
    q: list[float] = []
    for i, date in enumerate(dates):
        c = spy[i]
        q.append(c)
        sum50 += c
        sum200 += c
        if len(q) > 50:
            sum50 -= q[len(q) - 51]
        if len(q) > 200:
            sum200 -= q[len(q) - 201]
        regime = "n/a"
        dma50 = None
        dma200 = None
        if len(q) >= 200:
            dma50 = sum50 / 50
            dma200 = sum200 / 200
            v = vix[i]
            if v is not None and (v > 35 or (c < dma200 and v > 25)):
                regime = "crisis"
            elif c < dma200:
                regime = "bear"
            elif c > dma50 and dma50 > dma200:
                regime = "bull"
            else:
                regime = "sideways"
        out.append(
            {
                "date": date,
                "regime": regime,
                "vix": None if vix[i] is None else _to_fixed(vix[i], 2),
                "dma50": None if dma50 is None else _to_fixed(dma50, 2),
                "dma200": None if dma200 is None else _to_fixed(dma200, 2),
            }
        )
    return out


def regime_counts(regimes: list[dict]) -> dict:
    """Count days per regime label (includes "n/a")."""
    c = {"bull": 0, "bear": 0, "sideways": 0, "crisis": 0, "n/a": 0}
    for r in regimes:
        c[r["regime"]] += 1
    return c
