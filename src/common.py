"""Shared paths, data loading and plotting helpers."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "lung_cancer_incidence_1979_2022.csv"
SHP = ROOT / "data" / "external" / "county_boundary" / "COUNTY_MOI_1140318.shp"
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
TABLE = ROOT / "outputs" / "tables"

RATE_COL = "年齡標準化發生率  WHO 2000世界標準人口 (每10萬人口)"
LUNG = "肺、支氣管及氣管"
EXCLUDED = ["連江縣"]  # too few cases -> NaN / extreme jumps
RANDOM_STATE = 42
GROUP_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

_CJK_FONTS = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK TC",
              "Noto Sans TC", "Microsoft JhengHei", "Taipei Sans TC Beta"]


def setup_plot():
    """Pick the first installed CJK font so Chinese labels render."""
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    for d in (FIG, TABLE, PROCESSED):
        d.mkdir(parents=True, exist_ok=True)


def load_pivot() -> pd.DataFrame:
    """County x year matrix of lung-cancer age-standardised incidence (both sexes),
    Lienchiang excluded. Built by 01_preprocess.py."""
    return pd.read_csv(PROCESSED / "lung_age_std_county_year.csv", index_col=0).rename(columns=int)


def group_names(k: int):
    return [f"Group {chr(65 + i)}" for i in range(k)]

MOHW_DIR = ROOT / "data" / "raw" / "mohw_cause_of_death"
