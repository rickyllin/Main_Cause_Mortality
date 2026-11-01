"""EDA: trend lines, per-county boxplot, county-county correlation heatmap."""
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch

from common import load_pivot, setup_plot, figpath, tabpath

setup_plot()
pv = load_pivot()

fig, ax = plt.subplots(figsize=(12, 7))
for county, row in pv.iterrows():
    ax.plot(row.index, row.values, label=county, lw=1.4)
ax.set(title="肺癌年齡標準化發生率 (1979-2022)", xlabel="診斷年", ylabel="每10萬人口")
ax.legend(ncol=3, fontsize=8)
fig.savefig(figpath("eda_trend_lines.png"), dpi=200, bbox_inches="tight")

order = pv.T.median().sort_values().index
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=pv.T[order], ax=ax)
ax.set(title="各縣市肺癌年齡標準化發生率分布", ylabel="每10萬人口")
plt.setp(ax.get_xticklabels(), rotation=45)
fig.savefig(figpath("eda_boxplot_counties.png"), dpi=200, bbox_inches="tight")

# correlation between counties' 44-year trajectories (note: pv.T.corr(), not pv.corr())
corr = pv.T.corr()
leaves = sch.dendrogram(sch.linkage(corr, "ward"), no_plot=True)["leaves"]
corr = corr.iloc[leaves, leaves]
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5, ax=ax)
ax.set_title("縣市間相關係數 (Ward 排序)")
fig.savefig(figpath("eda_correlation_heatmap.png"), dpi=200, bbox_inches="tight")
corr.round(3).to_csv(tabpath("county_correlation.csv"), encoding="utf-8-sig")
