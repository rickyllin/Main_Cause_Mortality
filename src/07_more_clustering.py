"""Additional clustering families commonly used for longitudinal / trajectory data.

 shape/correlation  : hierarchical on 1-Pearson distance
 elastic (time-warp): DTW distance -> hierarchical (average) and k-medoids
 medoid             : PAM k-medoids (robust to the Kinmen/Penghu outliers)
 graph              : spectral clustering
 model-based        : Gaussian mixture on PCA scores (BIC); growth-curve mixture
                      (per-county quadratic trend coefficients -> GMM, a pragmatic stand-in for
                      latent-class growth / group-based trajectory modelling)
 soft               : fuzzy c-means (membership degrees)
 density            : DBSCAN (flags outlying trajectories as noise)
Validation: silhouette (Euclidean on Z), ARI against Ward k=5, and column-bootstrap stability."""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from common import FIG, RANDOM_STATE, TABLE, group_names, load_pivot, setup_plot

setup_plot()
pv = load_pivot()
names = pv.index.to_list()
Z = StandardScaler().fit_transform(pv)          # same input as 04_cluster.py
years = np.array(pv.columns, dtype=float)
rng = np.random.default_rng(RANDOM_STATE)


def relabel(lab):
    """Order labels by first appearance in `names` so Group A is stable & readable."""
    m = {}
    return np.array([m.setdefault(l, len(m)) for l in lab])


# ---------- building blocks ----------
def dtw(a, b, band=6):
    n = len(a); D = np.full((n + 1, n + 1), np.inf); D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(n, i + band) + 1):
            c = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return np.sqrt(D[n, n])


def dtw_matrix(X):
    n = len(X); M = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        M[i, j] = M[j, i] = dtw(X[i], X[j])
    return M


def pam(D, k, n_init=30):
    """k-medoids on a precomputed distance matrix (alternating / Voronoi iteration)."""
    n = len(D); best = (np.inf, None)
    for _ in range(n_init):
        med = rng.choice(n, k, replace=False)
        for _ in range(100):
            lab = D[:, med].argmin(1)
            new = np.array([np.where(lab == c)[0][D[np.ix_(lab == c, lab == c)].sum(1).argmin()]
                            if (lab == c).any() else med[c] for c in range(k)])
            if set(new) == set(med): break
            med = new
        cost = D[np.arange(n), med[D[:, med].argmin(1)]].sum()
        if cost < best[0]: best = (cost, D[:, med].argmin(1))
    return best[1]


def fuzzy_cmeans(X, k, m=2.0, iters=300):
    n = len(X); U = rng.dirichlet(np.ones(k), n)
    for _ in range(iters):
        W = U ** m; C = (W.T @ X) / W.sum(0)[:, None]
        d = np.linalg.norm(X[:, None] - C[None], axis=2) + 1e-12
        Un = 1 / ((d[:, :, None] / d[:, None, :]) ** (2 / (m - 1))).sum(2)
        if np.abs(Un - U).max() < 1e-6: U = Un; break
        U = Un
    return U


def growth_features(df):
    """Per-county quadratic trend on raw rates (time centred/scaled): level, slope, curvature."""
    t = (years - years.mean()) / years.std()
    F = np.array([np.polyfit(t, row, 2)[::-1] for row in df.values])   # [c0, c1, c2]
    return StandardScaler().fit_transform(F)


def gmm_bic(X, ks=range(1, 6), cov="spherical"):
    return {k: GaussianMixture(k, covariance_type=cov, n_init=10, random_state=RANDOM_STATE).fit(X).bic(X) for k in ks}


# ---------- distance matrices ----------
D_corr = squareform(pdist(Z, "correlation"))
D_dtw = dtw_matrix(Z)
scores4 = PCA(n_components=4).fit_transform(Z)
G = growth_features(pv)

# model selection for the mixtures
bic_pca, bic_growth = gmm_bic(scores4), gmm_bic(G)
pd.DataFrame({"GMM_PCA4": bic_pca, "GMM_growth": bic_growth}).round(1).to_csv(TABLE / "gmm_bic.csv")
print("BIC (PCA scores):", {k: round(v, 1) for k, v in bic_pca.items()})
print("BIC (growth coef):", {k: round(v, 1) for k, v in bic_growth.items()})


# ---------- method registry: name -> f(k) -> labels ----------
def hier_precomp(D, link):
    return lambda k: fcluster(linkage(squareform(D, checks=False), link), k, "maxclust") - 1


METHODS = {
    "KMeans":          lambda k: KMeans(k, n_init=10, random_state=RANDOM_STATE).fit_predict(Z),
    "Ward":            lambda k: AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z),
    "PAM_euclid":      lambda k: pam(squareform(pdist(Z)), k),
    "Hier_corr_avg":   hier_precomp(D_corr, "average"),
    "Hier_corr_comp":  hier_precomp(D_corr, "complete"),
    "Hier_DTW_avg":    hier_precomp(D_dtw, "average"),
    "PAM_DTW":         lambda k: pam(D_dtw, k),
    "Spectral_kNN":    lambda k: SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=5,
                                                    random_state=RANDOM_STATE).fit_predict(Z),
    "GMM_PCA4":        lambda k: GaussianMixture(k, covariance_type="spherical", n_init=10,
                                                 random_state=RANDOM_STATE).fit(scores4).predict(scores4),
    "GMM_growth":      lambda k: GaussianMixture(k, covariance_type="spherical", n_init=10,
                                                 random_state=RANDOM_STATE).fit(G).predict(G),
    "FuzzyCMeans":     lambda k: fuzzy_cmeans(Z, k).argmax(1),
}

# ---------- run k = 2..5 ----------
assign = pd.DataFrame(index=names)
sil = {}
for k in (2, 3, 4, 5):
    for m, f in METHODS.items():
        lab = relabel(np.asarray(f(k)))
        if len(set(lab)) < 2: continue
        assign[f"{m}_k{k}"] = [group_names(k)[l] if l < k else f"G{l}" for l in lab]
        sil[(m, k)] = silhouette_score(Z, lab) if len(set(lab)) < len(lab) else np.nan

# DBSCAN: density-based, k not specified; noise (-1) = trajectories unlike any dense group
nn3 = np.sort(squareform(pdist(Z)), axis=1)[:, 3]
eps = float(np.quantile(nn3, 0.8))
db = DBSCAN(eps=eps, min_samples=3).fit_predict(Z)
assign["DBSCAN"] = ["noise" if l < 0 else f"Cluster {l}" for l in db]
print(f"DBSCAN eps={eps:.2f}: noise =", [n for n, l in zip(names, db) if l < 0])

# fuzzy memberships (k=3,5) and growth-curve coefficients, for reporting
for k in (3, 5):
    U = fuzzy_cmeans(Z, k)
    pd.DataFrame(U, index=names, columns=group_names(k)).round(3).to_csv(
        TABLE / f"fuzzy_membership_k{k}.csv", encoding="utf-8-sig")
pd.DataFrame(G, index=names, columns=["level", "slope", "curvature"]).round(3).to_csv(
    TABLE / "growth_curve_features.csv", encoding="utf-8-sig")

assign.to_csv(TABLE / "more_cluster_assignments.csv", encoding="utf-8-sig")
S = pd.Series(sil).unstack(0).round(3)
S.to_csv(TABLE / "more_cluster_silhouette.csv")
print(S.T)

# ---------- agreement with the main analysis (ARI vs Ward k=5) ----------
ref = pd.read_csv(TABLE / "hclust_cluster_assignments.csv", index_col=0)
base = {k: ref[f"Ward_k{k}"] for k in (2, 3, 4, 5)}
ari_tab = pd.DataFrame({k: {m: ari(base[k], assign[f"{m}_k{k}"]) for m in METHODS if f"{m}_k{k}" in assign}
                        for k in (2, 3, 4, 5)}).round(3)
ari_tab.to_csv(TABLE / "more_cluster_ari_vs_ward.csv")
print("ARI vs Ward:\n", ari_tab)

# ---------- column-bootstrap stability (resample years with replacement) ----------
def stability(method, k, B=100):
    ref_lab = relabel(np.asarray(METHODS[method](k))); out = []
    for _ in range(B):
        idx = rng.integers(0, Z.shape[1], Z.shape[1])
        Zb = StandardScaler().fit_transform(pv.values[:, idx])
        lab = _fit_on(method, k, Zb)
        out.append(ari(ref_lab, lab))
    return np.mean(out), np.std(out)


def _fit_on(method, k, Zb):
    if method == "KMeans":  return KMeans(k, n_init=10, random_state=0).fit_predict(Zb)
    if method == "Ward":    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Zb)
    if method == "PAM_euclid": return pam(squareform(pdist(Zb)), k, n_init=10)
    if method == "Hier_corr_avg":
        return fcluster(linkage(pdist(Zb, "correlation"), "average"), k, "maxclust")
    if method == "Spectral_kNN":
        return SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=5, random_state=0).fit_predict(Zb)
    if method == "GMM_PCA4":
        s = PCA(n_components=4).fit_transform(Zb)
        return GaussianMixture(k, covariance_type="spherical", n_init=5, random_state=0).fit(s).predict(s)
    raise ValueError(method)


stab_rows = []
for m in ["KMeans", "Ward", "PAM_euclid", "Hier_corr_avg", "Spectral_kNN", "GMM_PCA4"]:
    for k in (2, 5):
        mu, sd = stability(m, k)
        stab_rows.append({"method": m, "k": k, "ARI_mean": round(mu, 3), "ARI_sd": round(sd, 3)})
stab = pd.DataFrame(stab_rows); stab.to_csv(TABLE / "more_cluster_bootstrap_stability.csv", index=False)
print(stab.pivot(index="method", columns="k", values="ARI_mean"))

# ---------- figures ----------
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
sns.heatmap(ari_tab, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax[0])
ax[0].set(title="與 Ward 分群之一致性 (ARI)", xlabel="k")
st = stab.pivot(index="method", columns="k", values="ARI_mean")
sns.heatmap(st, annot=True, fmt=".2f", cmap="YlOrRd_r", vmin=0, vmax=1, ax=ax[1])
ax[1].set(title="年度重抽樣穩定性 (bootstrap ARI)", xlabel="k")
fig.savefig(FIG / "more_cluster_agreement_stability.png", dpi=200, bbox_inches="tight")

# membership heatmap for k=5 across all methods (rows = counties ordered by Ward k=5)
order = ref.sort_values("Ward_k5").index
cols = [c for c in assign.columns if c.endswith("_k5")]
code = assign.loc[order, cols].apply(lambda s: s.astype("category").cat.codes)
fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(code, cmap="Set1", cbar=False, linewidths=.5, ax=ax,
            xticklabels=[c.replace("_k5", "") for c in cols], yticklabels=order)
ax.set_title("各演算法 k=5 分群歸屬（列：縣市；同色=同群，色彩編碼僅在欄內有意義）")
plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
fig.savefig(FIG / "more_cluster_membership_k5.png", dpi=200, bbox_inches="tight")

# fuzzy membership bar (k=3)
U3 = pd.read_csv(TABLE / "fuzzy_membership_k3.csv", index_col=0).loc[order]
U3.plot.barh(stacked=True, figsize=(9, 8), colormap="Set1", width=.85)
plt.title("Fuzzy c-means (k=3)：各縣市對各群之隸屬度"); plt.gca().invert_yaxis()
plt.savefig(FIG / "fuzzy_membership_k3.png", dpi=200, bbox_inches="tight")
