"""PCA on the county x year matrix: correlation-matrix (standardised) vs covariance-matrix."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import load_pivot, setup_plot, figpath, tabpath

setup_plot()
pv = load_pivot()
X = {"corr": StandardScaler().fit_transform(pv), "cov": pv.values - pv.values.mean(axis=0)}
fits = {k: PCA().fit(v) for k, v in X.items()}

summary = pd.DataFrame({f"{k}_ratio": p.explained_variance_ratio_ for k, p in fits.items()})
summary.index = np.arange(1, len(summary) + 1)
for k, p in fits.items():
    summary[f"{k}_cum"] = np.cumsum(p.explained_variance_ratio_)
summary.round(4).to_csv(tabpath("pca_explained_variance.csv"), encoding="utf-8-sig")
print(summary.head(7).map(lambda x: f"{x:.2%}"))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for k, p in fits.items():
    axes[0].plot(range(1, 11), p.explained_variance_[:10], "o--", label=k)
    axes[1].plot(range(1, 21), np.cumsum(p.explained_variance_ratio_)[:20], "o--", label=k)
axes[0].set(title="Scree plot (eigenvalue)", xlabel="PC")
axes[1].set(title="累積解釋變異", xlabel="PC")
axes[1].axhline(.8, color="r", ls=":")
for a in axes: a.legend(); a.grid(alpha=.3)
fig.savefig(figpath("pca_scree_cumulative.png"), dpi=200, bbox_inches="tight")

# loadings (years) and scores (counties) for the correlation-matrix PCA
p = fits["corr"]
pcs = [f"PC{i}" for i in range(1, 5)]
load = pd.DataFrame(p.components_[:4].T * np.sqrt(p.explained_variance_[:4]), index=pv.columns, columns=pcs)
score = pd.DataFrame(p.transform(X["corr"])[:, :4], index=pv.index, columns=pcs)
load.round(4).to_csv(tabpath("pca_corr_loadings_years.csv"), encoding="utf-8-sig")
score.round(4).to_csv(tabpath("pca_corr_scores_counties.csv"), encoding="utf-8-sig")

# biplot: counties = points, years = arrows
fig, ax = plt.subplots(figsize=(12, 10))
ax.scatter(score.PC1, score.PC2, c="tab:red", s=60)
for c, r in score.iterrows():
    ax.annotate(c, (r.PC1, r.PC2), xytext=(0, 7), textcoords="offset points", ha="center")
scale = np.abs(score[["PC1", "PC2"]].values).max() / np.abs(load[["PC1", "PC2"]].values).max() * .8
for y, r in load.iterrows():
    ax.arrow(0, 0, r.PC1 * scale, r.PC2 * scale, color="tab:blue", alpha=.35, head_width=.05)
    if y % 5 == 0 or y in (1979, 2022):
        ax.text(r.PC1 * scale * 1.08, r.PC2 * scale * 1.08, str(y), color="tab:blue", fontsize=9)
ax.axhline(0, c="grey", lw=.5); ax.axvline(0, c="grey", lw=.5)
ax.set(xlabel=f"PC1 ({p.explained_variance_ratio_[0]:.1%})", ylabel=f"PC2 ({p.explained_variance_ratio_[1]:.1%})",
       title="PCA biplot (correlation matrix)")
fig.savefig(figpath("pca_biplot_corr.png"), dpi=200, bbox_inches="tight")
