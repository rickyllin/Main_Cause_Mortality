"""Bayesian hierarchical (partial-pooling) model for county lung-cancer incidence trajectories.

Observation (log scale, delta-method sampling variance from the registry case count n):
    y_ct = log ASIR_ct ~ N( x_t' theta_c ,  phi / n_ct ),   x_t = (1, tau, tau^2)
Hierarchy:
    theta_c ~ N(mu, Sigma)  (c = 1..21),  mu ~ N(0, 10^2 I),  Sigma ~ Inv-Wishart(nu0, Psi0),
    phi ~ Inv-Gamma(0.01, 0.01)   (extra-Poisson dispersion; also absorbs age-standardisation variance)
Sampled with a Gibbs sampler (all full conditionals are conjugate).
Outputs: shrunken county levels/trends with 95% credible intervals, shrinkage factors,
posterior co-clustering probabilities (cluster the theta draws with K-means, k=5)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import invwishart
from sklearn.cluster import KMeans

from common import EXCLUDED, FIG, LUNG, RANDOM_STATE, RAW_CSV, TABLE, load_pivot, setup_plot

setup_plot()
rng = np.random.default_rng(RANDOM_STATE)

# ---- data: ASIR (matrix from 01) and the case counts (from the raw CSV) -------------------
pv = load_pivot(); names = pv.index.to_list(); years = np.array(pv.columns)
raw = pd.read_csv(RAW_CSV)
raw = raw[(raw["性別"] == "全") & (raw["縣市別"] != "全國") & (raw["癌症別"] == LUNG)].copy()
raw["n"] = pd.to_numeric(raw["癌症發生數"].astype(str).str.replace(",", "").str.strip())
N = raw.pivot_table(index="縣市別", columns="癌症診斷年", values="n", aggfunc="sum").loc[names, years]
assert (N.values > 0).all()
Y = np.log(pv.values); V = 1.0 / N.values                     # V = delta-method Var(log rate) / phi
tau = (years - years.mean()) / years.std()
X = np.column_stack([np.ones_like(tau), tau, tau ** 2]); P = X.shape[1]; J, T = Y.shape

# ---- Gibbs sampler -------------------------------------------------------------------------
nu0, Psi0 = P + 2, np.diag([0.3, 0.2, 0.1]) ** 2 * (P + 2)
theta = np.array([np.linalg.lstsq(X, y, rcond=None)[0] for y in Y]); mu = theta.mean(0)
Sig = np.cov(theta.T) + 1e-3 * np.eye(P); phi = 1.0
ITER, BURN, THIN = 6000, 1500, 5
keep = {"theta": [], "mu": [], "Sig": [], "phi": []}
for it in range(ITER):
    Sinv = np.linalg.inv(Sig)
    for c in range(J):
        W = 1.0 / (phi * V[c])
        prec = Sinv + (X * W[:, None]).T @ X
        cov = np.linalg.inv(prec)
        m = cov @ (Sinv @ mu + (X * W[:, None]).T @ Y[c])
        theta[c] = rng.multivariate_normal(m, cov)
    cov_mu = np.linalg.inv(J * Sinv + np.eye(P) / 100)
    mu = rng.multivariate_normal(cov_mu @ (Sinv @ theta.sum(0)), cov_mu)
    d = theta - mu
    Sig = invwishart.rvs(df=nu0 + J, scale=Psi0 + d.T @ d, random_state=rng)
    res = Y - theta @ X.T
    phi = 1.0 / rng.gamma(0.01 + J * T / 2, 1.0 / (0.01 + 0.5 * (res ** 2 / V).sum()))
    if it >= BURN and (it - BURN) % THIN == 0:
        for k, v in zip(keep, (theta.copy(), mu.copy(), Sig.copy(), phi)): keep[k].append(v)
TH = np.array(keep["theta"]); PHI = np.array(keep["phi"]); MU = np.array(keep["mu"]); SIG = np.array(keep["Sig"])
print(f"{len(TH)} draws | phi = {PHI.mean():.2f} [{np.quantile(PHI, .025):.2f}, {np.quantile(PHI, .975):.2f}]")
print("population mean (level, slope, curv):", MU.mean(0).round(3), "| between-county SD:", np.sqrt(SIG.mean(0).diagonal()).round(3))

# ---- posterior summaries ---------------------------------------------------------------------
curve = np.exp(np.einsum("dcp,tp->dct", TH, X))              # (draws, county, year) ASIR curves
q = lambda a, p: np.quantile(a, p, axis=0)
last = list(years).index(2022); win = [i for i, y in enumerate(years) if 2018 <= y <= 2022]
lvl22 = curve[:, :, last]; lvl5 = curve[:, :, win].mean(2)
raw22 = pv.loc[:, 2018:2022].mean(1).values
rank5 = (-lvl5).argsort(1).argsort(1) + 1                    # 1 = highest
growth = np.exp(TH[:, :, 1] / years.std()) - 1                # relative change per year at mid-period (b/sd)
summ = pd.DataFrame({
    "raw_2018_22": raw22, "post_median_2018_22": q(lvl5, .5), "lo95": q(lvl5, .025), "hi95": q(lvl5, .975),
    "ci_width": q(lvl5, .975) - q(lvl5, .025), "rank_median": np.median(rank5, 0),
    "P_top5": (rank5 <= 5).mean(0), "P_above_national_median": (lvl5 > np.median(lvl5, 1)[:, None]).mean(0),
    "growth_pct_per_yr_mid": 100 * np.median(growth, 0)}, index=names)
summ["shrink_pct"] = 100 * (summ.post_median_2018_22 - summ.raw_2018_22) / summ.raw_2018_22
summ.round(3).sort_values("post_median_2018_22").to_csv(TABLE / "bayes_county_summary.csv", encoding="utf-8-sig")
print(summ.round(2).sort_values("post_median_2018_22"))
pd.DataFrame({"mu_level": MU[:, 0], "mu_slope": MU[:, 1], "mu_curv": MU[:, 2], "phi": PHI}).describe().round(3).to_csv(
    TABLE / "bayes_hyperparameters.csv")

# ---- figures -----------------------------------------------------------------------------------
order = summ.sort_values("post_median_2018_22").index
fig, ax = plt.subplots(figsize=(9, 8)); y = np.arange(len(order))
s = summ.loc[order]
ax.errorbar(s.post_median_2018_22, y, xerr=[s.post_median_2018_22 - s.lo95, s.hi95 - s.post_median_2018_22],
            fmt="o", c="tab:blue", capsize=3, label="貝氏後驗中位數與95%可信區間")
ax.scatter(s.raw_2018_22, y, marker="x", c="tab:red", zorder=3, label="原始五年平均")
ax.set_yticks(y); ax.set_yticklabels(order); ax.legend(); ax.grid(alpha=.3)
ax.set(xlabel="肺癌年齡標準化發生率 (2018–22, /10萬)", title="階層模型之部分合併估計（小區域估計）")
fig.savefig(FIG / "bayes_forest_2018_22.png", dpi=200, bbox_inches="tight")

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
for ax, c in zip(axes.ravel(), ["澎湖縣", "金門縣", "台北市", "台南市", "新北市", "苗栗縣"]):
    i = names.index(c)
    ax.fill_between(years, q(curve[:, i], .025), q(curve[:, i], .975), alpha=.25, color="tab:blue")
    ax.plot(years, q(curve[:, i], .5), c="tab:blue", label="後驗中位數"); ax.scatter(years, pv.iloc[i], s=14, c="tab:red", label="原始")
    ax.set_title(c)
axes[0, 0].legend(); fig.suptitle("貝氏階層成長曲線 vs. 原始年度值（95%可信帶）")
fig.savefig(FIG / "bayes_trajectories.png", dpi=200, bbox_inches="tight")

# ---- uncertainty-aware clustering: co-clustering probability of theta draws --------------------
sc = TH.reshape(-1, P).std(0)
sel = rng.choice(len(TH), 300, replace=False); co = np.zeros((J, J))
for d in sel:
    lab = KMeans(5, n_init=3, random_state=0).fit_predict(TH[d] / sc)
    co += lab[:, None] == lab[None]
co /= len(sel)
ref = pd.read_csv(TABLE / "hclust_cluster_assignments.csv", index_col=0).loc[names]
ordr = ref.sort_values("Ward_k5").index
cop = pd.DataFrame(co, index=names, columns=names).loc[ordr, ordr]
cop.round(3).to_csv(TABLE / "bayes_coclustering_prob_k5.csv", encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(cop, cmap="viridis", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "同群機率"})
ax.set_title("後驗同群機率（對 θ 的後驗抽樣做 K-means, k=5；列依 Ward k=5 排序）")
fig.savefig(FIG / "bayes_coclustering_k5.png", dpi=200, bbox_inches="tight")
print("min within-Ward-group co-cluster prob by group:",
      {g: round(cop.loc[idx.index, idx.index].values.min(), 2) for g, idx in ref.groupby("Ward_k5")})
