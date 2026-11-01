"""Lung-cancer (ICD-10 C33-C34) age-standardised MORTALITY by county from MOHW cause-of-death
tables (民國109-112, 114 => 2020-2023, 2025; 113 is not in the data folder), and the
mortality-to-incidence ratio (MIR) against registry incidence for the overlapping years 2020-2022."""
import re

import matplotlib.pyplot as plt
import pandas as pd

from common import EXCLUDED, MOHW_DIR, load_pivot, setup_plot, figpath, tabpath

setup_plot()
# column positions on sheet "<county>2" (表2): total std=5, male std=12, female std=18
COLS = {"deaths": 3, "crude": 4, "asmr": 5, "asmr_male": 12, "asmr_female": 18}
rows = []
for f in sorted(MOHW_DIR.glob("roc*_county.xls")):
    roc = int(re.search(r"roc(\d+)_", f.name).group(1))
    xl = pd.ExcelFile(f)
    for sheet in (s for s in xl.sheet_names if s.endswith("2") and not s.startswith("全國")):
        d = xl.parse(sheet, header=None)
        hit = d[d[1].astype(str).str.strip() == "C33-C34"]
        if hit.empty:
            continue
        r = hit.iloc[0]
        rows.append({"縣市別": sheet[:-1].replace("臺", "台"), "year": roc + 1911,
                     **{k: pd.to_numeric(r[c], errors="coerce") for k, c in COLS.items()}})
mort = pd.DataFrame(rows)
mort = mort[~mort["縣市別"].isin(EXCLUDED)]
mort.to_csv(tabpath("lung_mortality_county_year.csv"), index=False, encoding="utf-8-sig")
print(mort.groupby("year").size().to_dict())

inc = load_pivot().stack().rename("asir").reset_index()
inc.columns = ["縣市別", "year", "asir"]
m = mort.merge(inc, on=["縣市別", "year"])
m["MIR"] = m["asmr"] / m["asir"]  # age-standardised mortality / age-standardised incidence
m.to_csv(tabpath("lung_mir_county_year.csv"), index=False, encoding="utf-8-sig")

avg = m.groupby("縣市別")[["asir", "asmr", "MIR"]].mean().sort_values("MIR")
avg.round(3).to_csv(tabpath("lung_mir_county_mean_2020_2022.csv"), encoding="utf-8-sig")
print(avg.round(2))
print("corr(ASIR,ASMR) by year:", m.groupby("year").apply(lambda g: g.asir.corr(g.asmr)).round(2).to_dict())

fig, ax = plt.subplots(figsize=(9, 8))
ax.scatter(avg.asir, avg.asmr, s=60)
for c, r in avg.iterrows():
    ax.annotate(c, (r.asir, r.asmr), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9)
lim = max(avg.asir.max(), avg.asmr.max()) * 1.05
for mir in (0.4, 0.6, 0.8):
    ax.plot([0, lim], [0, lim * mir], "--", c="grey", lw=.7); ax.text(lim, lim * mir, f"MIR={mir}", fontsize=8)
ax.set(xlabel="年齡標準化發生率 (2020–22 平均, /10萬)", ylabel="年齡標準化死亡率 (2020–22 平均, /10萬)",
       title="肺癌：發生率 vs 死亡率（各縣市）")
fig.savefig(figpath("incidence_vs_mortality_scatter.png"), dpi=200, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(10, 5.5))
mort.pivot(index="縣市別", columns="year", values="asmr").loc[avg.index].plot.bar(ax=ax)
ax.set(title="肺癌年齡標準化死亡率（衛福部死因統計）", ylabel="每10萬人口"); plt.setp(ax.get_xticklabels(), rotation=45)
fig.savefig(figpath("mortality_by_county_year.png"), dpi=200, bbox_inches="tight")
