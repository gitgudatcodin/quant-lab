"""Cached data loaders for the Quant Strategy Lab.

All JSONs live in quantlab/data/ (copied from the original HTML build dirs).
Loaders use st.cache_data so data is read once per session.
"""
from __future__ import annotations

import json
import os

import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load(name: str):
    with open(os.path.join(DATA_DIR, name), "r") as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading pairs price data…")
def load_pairs_daily():
    """daily.json: {dates:[str], px:{SYM:[float]}, spy:[float], sectors:{SYM:str}, meta:{…}}.
    150 liquid S&P names, daily 2018-01-02 → 2026-08-31."""
    return _load("daily.json")


@st.cache_data(show_spinner="Loading ML feature panel…")
def load_ml_features():
    """features.json: {months:[ym], fnames:[20], symbols:[…],
    feats:{SYM:[[20]|null per month]}, nextRet:{SYM:[bp int|null]}, spyRet:[bp int]}."""
    return _load("features.json")


@st.cache_data(show_spinner="Loading SPY daily data…")
def load_spy_daily():
    """spy_daily.json: {dates:[str], px:[float]} daily SPY 2015→2026."""
    return _load("spy_daily.json")


@st.cache_data(show_spinner="Loading sentiment data…")
def load_sentiment_data():
    """sentiment_data.json: {dates, spy, vix, prices:{5 sample tickers}} 2018→2026."""
    return _load("sentiment_data.json")


@st.cache_data(show_spinner="Loading lexicon…")
def load_lexicon():
    """lexicon.json: {pos:[…], neg:[…], unc:[…], lit:[…]} Loughran-McDonald style."""
    return _load("lexicon.json")


@st.cache_data(show_spinner="Loading sample earnings events…")
def load_samples():
    """samples.json: 5 labeled fictional earnings press-release snippets."""
    return _load("samples.json")


@st.cache_data(show_spinner="Loading earnings-event panel…")
def load_events():
    """events.json: [{ticker, fileDate, entryDate, score, pos, neg, total}];
    2,515 scored earnings events, 90 names, 2019→2026. Entry = next trading
    day's close after the SEC filing date (strictly point-in-time)."""
    return _load("events.json")


@st.cache_data(show_spinner="Loading sentiment-strategy prices…")
def load_sentstrat_prices():
    """px100.json: {dates, px:{SYM:[float]}, spy:[float], sectors} daily 100-name panel."""
    return _load("px100.json")
