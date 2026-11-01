"""K-means (elbow/silhouette, PCA scatter) and hierarchical clustering (4 linkages, k=2..5)."""
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from common import GROUP_COLORS, RANDOM_STATE, group_names, load_pivot, setup_plot, figpath, tabpath

setup_plot()
pv = load_pivot()
Xs = StandardScaler().fit_transform(pv)  # standardise each year across counties
coords = PCA(n_components=2).fit_transform(Xs)
K_RANGE = range(2, 11)

# ---- K-means ---------------------------------------------------------------
km = lambda k: KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(Xs)
sse = {k: km(k).inertia_ for k in range(1, 11)}
sil = {k: silhouette_score(Xs, km(k).labels_) for k in K_RANGE}
pd.DataFrame({"SSE": sse, "silhouette": pd.Series(sil)}).to_csv(tabpath("kmeans_selection.csv"))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(list(sse), list(sse.values()), "o-"); ax[0].set(title="Elbow (SSE)", xlabel="k")
ax[1].plot(list(sil), list(sil.values()), "o-"); ax[1].set(title="Silhouette", xlabel="k")
fig.savefig(figpath("kmeans_elbow_silhouette.png"), dpi=200, bbox_inches="tight")

km_labels = pd.DataFrame(index=pv.index)
for k in (2, 3, 4, 5):
    km_labels[f"KMeans_k{k}"] = [group_names(k)[l] for l in km(k).labels_]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(x=coords[:, 0], y=-coords[:, 1], hue=km_labels[f"KMeans_k{k}"].values,
                    hue_order=group_names(k), palette=GROUP_COLORS[:k], s=180, edgecolor="k", ax=ax)
    for name, (x, y) in zip(pv.index, coords):
        ax.annotate(name, (x, -y), xytext=(0, 10), textcoords="offset points", ha="center")
    ax.set(title=f"K-means (k={k}) on PC1/PC2", xlabel="PC1", ylabel="PC2")
    fig.savefig(figpath(f"kmeans_pca_k{k}.png"), dpi=200, bbox_inches="tight")
km_labels.to_csv(tabpath("kmeans_cluster_assignments.csv"), encoding="utf-8-sig")

# ---- Hierarchical ----------------------------------------------------------
METHODS = ["single", "complete", "average", "ward"]
rows = [{"k": k, "Method": m.capitalize(),
         "Silhouette": silhouette_score(Xs, AgglomerativeClustering(n_clusters=k, linkage=m).fit_predict(Xs))}
        for k in K_RANGE for m in METHODS]
sc = pd.DataFrame(rows)
sc.pivot(index="k", columns="Method", values="Silhouette").round(4).to_csv(tabpath("hclust_silhouette.csv"))
fig, ax = plt.subplots(figsize=(11, 5.5))
sns.lineplot(data=sc, x="k", y="Silhouette", hue="Method", marker="o", ax=ax)
ax.set(title="Silhouette score by linkage"); ax.grid(alpha=.3)
fig.savefig(figpath("hclust_silhouette.png"), dpi=200, bbox_inches="tight")

hc = pd.DataFrame(index=pv.index)
for k in (2, 3, 4, 5):
    for m in METHODS:
        lab = AgglomerativeClustering(n_clusters=k, linkage=m).fit_predict(Xs)
        hc[f"{m.capitalize()}_k{k}"] = [group_names(k)[l] for l in lab]
hc.to_csv(tabpath("hclust_cluster_assignments.csv"), encoding="utf-8-sig")

for m in METHODS:
    Z = sch.linkage(Xs, method=m)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sch.dendrogram(Z, labels=pv.index.tolist(), ax=ax, color_threshold=0.7 * Z[:, 2].max())
    ax.set(title=f"Hierarchical clustering ({m} linkage, Euclidean)", ylabel="Distance")
    fig.savefig(figpath(f"dendrogram_{m}.png"), dpi=200, bbox_inches="tight")
print("K-means silhouette:", {k: round(v, 3) for k, v in sil.items()})
print(sc.pivot(index="k", columns="Method", values="Silhouette").round(3))
