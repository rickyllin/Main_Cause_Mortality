"""Choropleth maps of the cluster assignments + cluster-mean trend lines."""
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from common import EXCLUDED, FIG, GROUP_COLORS, SHP, TABLE, group_names, load_pivot, setup_plot

setup_plot()
pv = load_pivot()
gdf = gpd.read_file(SHP)
gdf["County"] = gdf["COUNTYNAME"].str.replace("臺", "台")
gdf = gdf[~gdf["County"].isin(EXCLUDED)]

assign = pd.concat([pd.read_csv(TABLE / "kmeans_cluster_assignments.csv", index_col=0),
                    pd.read_csv(TABLE / "hclust_cluster_assignments.csv", index_col=0)], axis=1)
TARGETS = ["KMeans_k2", "KMeans_k5", "Ward_k2", "Ward_k5", "Average_k4", "Complete_k4", "Single_k4"]

for col in TARGETS:
    k = assign[col].nunique()
    cmap = dict(zip(group_names(k), GROUP_COLORS))
    d = gdf.merge(assign[[col]], left_on="County", right_index=True, how="left")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [1, 1.2]})
    # main island window (drop Kinmen/Penghu offshore extents from the main frame)
    d.plot(ax=ax, color=d[col].map(cmap).fillna("#cbd5e1"), edgecolor="k", lw=.5)
    ax.set_xlim(119.9, 122.1); ax.set_ylim(21.8, 25.4); ax.axis("off")
    ax.set_title(f"{col} 地理分布")
    for g in group_names(k):  # cluster-mean trends
        members = assign.index[assign[col] == g]
        ax2.plot(pv.columns, pv.loc[members].mean(), color=cmap[g], lw=2.5,
                 label=f"{g}: {'、'.join(members)}")
    ax2.set(title="群組平均肺癌發生率", xlabel="診斷年", ylabel="每10萬人口")
    ax2.legend(fontsize=8, loc="upper left"); ax2.grid(alpha=.3)
    fig.savefig(FIG / f"map_trend_{col}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
