"""Shared helpers for the health-economics case study (src/health_economics/).

Re-uses the project-wide settings in ../common.py but writes to its own output tree:
    outputs/health_economics/figures/<fe|did|causal|inequality>/   and   outputs/health_economics/tables/<...>/
"""
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402
from common import (EXCLUDED, LUNG, PROCESSED, RANDOM_STATE, RAW_CSV, ROOT, SHP, setup_plot)  # noqa: E402,F401

HE_OUT = ROOT / "outputs" / "health_economics"
PANEL = PROCESSED / "panel_lung_county_year.csv"
NATIONAL = PROCESSED / "panel_cancer_national_year.csv"
_TOPICS = [(r"^fe_", "fe"), (r"^did_", "did"), (r"^(sc_|dml_|sens_)", "causal"), (r"^(ineq_|sar_)", "inequality")]


def _topic(name):
    for pat, t in _TOPICS:
        if re.match(pat, name):
            return t
    raise ValueError(f"no output topic for {name!r}")


def figpath(name):
    d = HE_OUT / "figures" / _topic(name); d.mkdir(parents=True, exist_ok=True); return d / name


def tabpath(name):
    d = HE_OUT / "tables" / _topic(name); d.mkdir(parents=True, exist_ok=True); return d / name


def load_panel() -> pd.DataFrame:
    """Long county-year panel (21 counties x 44 years): asir, n (cases), crude, pop, ln_asir, ln_pop, oral_asir ..."""
    return pd.read_csv(PANEL)
