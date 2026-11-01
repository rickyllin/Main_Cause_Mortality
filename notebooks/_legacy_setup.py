"""Shared set-up for the notebooks in this folder.

The original Colab notebook began with cells that only work on Google Colab (Drive mount, font download,
GDAL). Those cells are replaced by `from _legacy_setup import *`, which provides the same variables
from the files in this repository:

    df, df_filtered              raw registry table (English column names) and the lung/both-sex/county subset
    pivot_table_age_std          county x year ASIR matrix (Lienchiang still included, 11 NaN cells)
    pivot_table_cleaned          same without 連江縣 (21 x 44)
    scaled_data                  pivot_table_cleaned standardised per year (columns)
    taiwan_map, county_col       county boundaries (EPSG:4326) and the name column
    font_name, prop              a CJK-capable font for matplotlib
The module lives in notebooks/ (one level above the notebook folders).
Legacy cells write PNG/CSV files to the working directory, which is set to notebooks/_scratch/ (git-ignored).
"""
import os
import warnings
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_here = Path.cwd().resolve()
ROOT = next(p for p in [_here, *_here.parents] if (p / "data" / "raw").exists() and (p / "src").exists())
CSV_PATH = ROOT / "data" / "raw" / "lung_cancer_incidence_1979_2022.csv"
SHP_PATH = ROOT / "data" / "external" / "county_boundary" / "COUNTY_MOI_1140318.shp"
SCRATCH = ROOT / "notebooks" / "_scratch"
SCRATCH.mkdir(exist_ok=True)
os.chdir(SCRATCH)

# --- font ---------------------------------------------------------------------------------------
_installed = {f.name for f in fm.fontManager.ttflist}
font_name = next((n for n in ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK TC", "Noto Sans TC",
                              "Microsoft JhengHei", "Taipei Sans TC Beta"] if n in _installed), "sans-serif")
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(font=font_name, style="whitegrid")
prop = fm.FontProperties(family=font_name)

# --- data ---------------------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
df.columns = ["Diagnosis_Year", "Gender", "City_County", "Cancer_Type", "Age_Standardized_Incidence_Rate",
              "Cancer_Incidence_Count", "Mean_Age", "Median_Age", "Crude_Incidence_Rate"]
df_filtered = df[(df["Gender"] == "全") & (df["City_County"] != "全國") & (df["Cancer_Type"] == "肺、支氣管及氣管")]
pivot_table_age_std = df_filtered.pivot_table(index="City_County", columns="Diagnosis_Year",
                                              values="Age_Standardized_Incidence_Rate", aggfunc="mean")
pivot_table_cleaned = pivot_table_age_std.drop(index="連江縣")
scaled_data = StandardScaler().fit_transform(pivot_table_cleaned)

# --- map ----------------------------------------------------------------------------------------
try:
    import geopandas as gpd
    taiwan_map = gpd.read_file(SHP_PATH).to_crs(epsg=4326)
    county_col = "COUNTYNAME"
except Exception as e:  # geopandas not installed -> only the map notebook needs it
    taiwan_map, county_col = None, "COUNTYNAME"
    print("geopandas unavailable:", e)
