"""Headless exercise of pages/1_Pairs_Trading.py with a stubbed streamlit.

Runs the module twice: (1) Scan button True -> exercises cached_scan,
IS-Sharpe loop, candidate table; (2) Backtest button True -> exercises
backtest_portfolio, plotly chart, metrics, trade table, CSV download.
Catches NameError/TypeError/KeyError etc. in all module-level branches.
"""
import os
import sys
import types
from contextlib import contextmanager
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

calls = {"plotly": 0, "dataframe": 0, "metric": 0, "download": 0}


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return lambda *a, **k: None


class _Col(_Ctx):
    def metric(self, *a, **k):
        calls["metric"] += 1
        return None


class _SessionState(dict):
    pass


class _Sidebar(_Ctx):
    def slider(self, label, lo, hi, default, step):
        return default

    def selectbox(self, label, options, index=0):
        return options[index]

    def date_input(self, label, default):
        return default

    def button(self, label, **k):
        return BUTTONS.pop(0) if BUTTONS else False

    def divider(self):
        return None

    def header(self, *a, **k):
        return None


class _St(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = _SessionState()
        self.sidebar = _Sidebar()

    def set_page_config(self, **k):
        return None

    def cache_data(self, **k):
        def deco(fn):
            return fn
        return deco

    def title(self, *a, **k):
        return None

    def header(self, *a, **k):
        return None

    def subheader(self, *a, **k):
        return None

    def markdown(self, *a, **k):
        return None

    def caption(self, *a, **k):
        return None

    def info(self, *a, **k):
        return None

    def warning(self, *a, **k):
        return None

    def success(self, *a, **k):
        return None

    def dataframe(self, *a, **k):
        calls["dataframe"] += 1
        return None

    def metric(self, *a, **k):
        calls["metric"] += 1
        return None

    def columns(self, n):
        return [_Col() for _ in range(n)]

    def progress(self, *a, **k):
        return _Ctx()

    @contextmanager
    def spinner(self, *a, **k):
        yield

    @contextmanager
    def expander(self, *a, **k):
        yield

    def empty(self):
        return _Ctx()

    def plotly_chart(self, fig, **k):
        calls["plotly"] += 1
        assert len(fig.data) == 2, "expected strategy + SPY traces"
        return None

    def download_button(self, *a, **k):
        calls["download"] += 1
        assert k.get("file_name") == "pairs_trades.csv"
        return None


st = _St()
sys.modules["streamlit"] = st
# stub plotly: use the real one (installed in venv)

BUTTONS = []  # consumed in order by sidebar.button


def run_page(scan, backtest):
    global BUTTONS
    BUTTONS = [scan, backtest]
    for mod in [m for m in sys.modules if m.startswith("page_under_test")]:
        del sys.modules[mod]
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "page_under_test", "pages/1_Pairs_Trading.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["page_under_test"] = mod
    spec.loader.exec_module(mod)


print("== pass 1: click Scan ==")
run_page(True, False)
rows = st.session_state.get("pairs_rows")
assert rows, "no rows after scan"
assert all("isSharpe" in r for r in rows), "IS Sharpe missing"
print(f"   rows={len(rows)} top={rows[0]['a']}/{rows[0]['b']} "
      f"isSharpe={rows[0]['isSharpe']:.2f}")

print("== pass 2: click Backtest ==")
run_page(False, True)
print("   plotly charts:", calls["plotly"], "| dataframes:", calls["dataframe"],
      "| metrics:", calls["metric"], "| downloads:", calls["download"])
assert calls["plotly"] == 1 and calls["metric"] == 5 and calls["download"] == 1
print("\nPAGE HEADLESS TEST PASSED")
