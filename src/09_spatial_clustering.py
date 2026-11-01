"""Spatially constrained clustering (only adjacent counties may merge) + spatial autocorrelation.

Adjacency: queen contiguity from the county shapefile; island counties (no land border) are
linked to their nearest mainland-or-island neighbours so the graph is connected.
Constrained clustering: Ward agglomeration restricted by the adjacency graph
(sklearn AgglomerativeClustering(connectivity=...)). Compared with the unconstrained Ward result."""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari, silhouette_score
from sklearn.preprocessing import StandardScaler

from common import EXCLUDED, GROUP_COLORS, RANDOM_STATE, SHP, group_names, load_pivot, setup_plot, figpath, tabpath

setup_plot()
rng = np.random.default_rng(RANDOM_STATE)
pv = load_pivot(); names = pv.index.to_list(); n = len(names)
Z = StandardScaler().fit_transform(pv)

# ---- adjacency -----------------------------------------------------------------------------
gdf = gpd.read_file(SHP).to_crs(3826)
gdf["County"] = gdf["COUNTYNAME"].str.replace("臺", "台")
gdf = gdf[~gdf["County"].isin(EXCLUDED)].set_index("County").loc[names]
geom = gdf.geometry
A = np.zeros((n, n), dtype=int)
for i in range(n):
    for j in range(i + 1, n):
        if geom.iloc[i].touches(geom.iloc[j]) or geom.iloc[i].intersects(geom.iloc[j]):
            A[i, j] = A[j, i] = 1
islands = [i for i in range(n) if A[i].sum() == 0]
for i in islands:                                    # link island(s) to nearest 2 other counties
    d = np.array([geom.iloc[i].distance(geom.iloc[j]) if j != i else np.inf for j in range(n)])
    for j in d.argsort()[:2]:
        A[i, j] = A[j, i] = 1
print("island links:", {names[i]: [names[j] for j in np.where(A[i])[0]] for i in islands})
assert connected_components(csr_matrix(A))[0] == 1
pd.DataFrame(A, index=names, columns=names).to_csv(tabpath("county_adjacency.csv"), encoding="utf-8-sig")

# ---- spatial autocorrelation (global Moran's I, row-standardised weights, permutation test) -
W = A / A.sum(1, keepdims=True)
def moran(x):
    z = x - x.mean(); return n * (z @ W @ z) / (z @ z) / W.sum()
def moran_p(x, B=9999):
    obs = moran(x); sims = np.array([moran(rng.permutation(x)) for _ in range(B)])
    return obs, (1 + (sims >= obs).sum()) / (B + 1)

pcs = PCA(n_components=2).fit_transform(Z)
feat = {"ASIR 1979–83": pv.loc[:, 1979:1983].mean(1).values, "ASIR 2018–22": pv.loc[:, 2018:2022].mean(1).values,
        "增幅(2018–22 − 1979–83)": (pv.loc[:, 2018:2022].mean(1) - pv.loc[:, 1979:1983].mean(1)).values,
        "PC1 得分": pcs[:, 0], "PC2 得分": pcs[:, 1]}
mo = pd.DataFrame([{"variable": k, "Moran_I": (r := moran_p(v))[0], "p_perm": r[1]} for k, v in feat.items()])
mo.round(4).to_csv(tabpath("moran_i.csv"), index=False, encoding="utf-8-sig"); print(mo.round(3))
# year-by-year Moran's I of the (raw) rates
yr = pd.Series({y: moran(pv[y].values) for y in pv.columns}); yr.round(4).to_csv(tabpath("moran_i_by_year.csv"), header=["Moran_I"])
print("Moran I by year: mean %.2f, 1979-89 mean %.2f, 2010-22 mean %.2f" % (yr.mean(), yr.loc[1979:1989].mean(), yr.loc[2010:2022].mean()))

# ---- constrained vs unconstrained Ward -------------------------------------------------------
conn = csr_matrix(A)
rows, assign = [], pd.DataFrame(index=names)
ref = pd.read_csv(tabpath("hclust_cluster_assignments.csv"), index_col=0).loc[names]
def n_components(lab):
    return sum(connected_components(csr_matrix(A[np.ix_(lab == c, lab == c)]))[0] for c in np.unique(lab))
for k in (2, 3, 4, 5, 6):
    sp = AgglomerativeClustering(n_clusters=k, linkage="ward", connectivity=conn).fit_predict(Z)
    un = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
    assign[f"SpatialWard_k{k}"] = [group_names(k)[l] for l in sp]
    rows.append({"k": k, "silhouette_spatial": silhouette_score(Z, sp), "silhouette_unconstrained": silhouette_score(Z, un),
                 "ARI_vs_unconstrained": ari(sp, un),
                 "pieces_unconstrained(=k means contiguous)": n_components(un), "pieces_spatial": n_components(sp)})
cmp = pd.DataFrame(rows).round(3); cmp.to_csv(tabpath("spatial_vs_unconstrained.csv"), index=False)
assign.to_csv(tabpath("spatial_cluster_assignments.csv"), encoding="utf-8-sig"); print(cmp.to_string(index=False))
for k in (3, 4, 5, 6):
    print(f"SpatialWard k={k}:", {g: list(x.index) for g, x in assign.groupby(f"SpatialWard_k{k}")})

# ---- maps + group trajectories --------------------------------------------------------------
for k in (3, 4, 5):
    col = f"SpatialWard_k{k}"; cmap = dict(zip(group_names(k), GROUP_COLORS))
    d = gdf.copy(); d["color"] = [cmap[g] for g in assign[col]]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [1, 1.2]})
    d.plot(ax=ax, color=d["color"], edgecolor="k", lw=.5)
    parts = d.drop(index="金門縣").explode(index_parts=False)
    x0, y0, x1, y1 = parts[parts.area > 30e6].total_bounds  # frame main island + Penghu; ignore far-off islets; Kinmen off-frame
    ax.set_xlim(x0 - 1e4, x1 + 1e4); ax.set_ylim(y0 - 1e4, y1 + 1e4); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"空間約束 Ward（k={k}；金門縣在圖框外，歸 {assign.loc['金門縣', col]}）")
    for g in group_names(k):
        mem = assign.index[assign[col] == g]
        ax2.plot(pv.columns, pv.loc[mem].mean(), color=cmap[g], lw=2.5, label=f"{g}: {'、'.join(mem)}")
    ax2.set(title="群組平均肺癌發生率", xlabel="診斷年", ylabel="每10萬人口"); ax2.legend(fontsize=8, loc="upper left"); ax2.grid(alpha=.3)
    fig.savefig(figpath(f"spatial_ward_map_k{k}.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5)); yr.plot(ax=ax, marker="o", ms=3)
ax.axhline(0, c="grey", lw=.6); ax.set(title="各年度肺癌發生率的空間自相關 (Moran's I)", xlabel="診斷年", ylabel="Moran's I"); ax.grid(alpha=.3)
fig.savefig(figpath("moran_i_by_year.png"), dpi=200, bbox_inches="tight")
