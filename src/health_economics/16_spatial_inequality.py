"""Health inequality and spatial autoregressive (SAR / SEM) models for county lung-cancer incidence.

Part 1  Inequality indices by year (21 counties): absolute (SD, range, population-weighted SII w.r.t. population rank) and relative
        (CV, Gini, Theil-T, RII, concentration index) measures.  County population rank is used as an urban-scale gradient because no
        income/education data are available; this is a descriptive gradient, not a socio-economic ranking.
Part 2  Spatial econometrics on 5-year windows (cross-sections of ln ASIR; n = 21; row-standardised queen + island-link weights):
        OLS, spatial lag (SAR) and spatial error (SEM) by maximum likelihood (concentrated likelihood over rho / lambda),
        LM-lag / LM-error / robust LM tests (Anselin et al. 1996), Moran's I of residuals; covariates = ln population (+ ln oral-cancer
        ASIR as a tobacco/betel risk proxy).  Robustness: 4-nearest-neighbour weights.
Part 3  Spatial beta-convergence: growth 1979-83 -> 2018-22 on initial level with a spatially lagged dependent variable."""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

from he_common import EXCLUDED, SHP, figpath, load_panel, setup_plot, tabpath

setup_plot()
rng = np.random.default_rng(42)
d = load_panel()
names = sorted(d.county.unique())
pv = d.pivot(index="county", columns="year", values="asir").loc[names]
pop = d.pivot(index="county", columns="year", values="pop").loc[names]

# ---------------- weights ----------------
adj = pd.read_csv(tabpath.__globals__["HE_OUT"].parent / "tables" / "spatial" / "county_adjacency.csv", index_col=0).loc[names, names].values.astype(float)
rs = lambda A: A / A.sum(1, keepdims=True)
W_q = rs(adj)
g = gpd.read_file(SHP).to_crs(3826); g["c"] = g.COUNTYNAME.str.replace("臺", "台"); g = g[~g.c.isin(EXCLUDED)].set_index("c").loc[names]
cen = np.column_stack([g.geometry.centroid.x, g.geometry.centroid.y])
D = np.linalg.norm(cen[:, None] - cen[None], axis=2); np.fill_diagonal(D, np.inf)
knn = np.zeros_like(D)
for i in range(len(names)): knn[i, np.argsort(D[i])[:4]] = 1
W_k = rs(knn)                      # asymmetric 4-NN (row-standardised)
print("queen+island: mean neighbours %.1f | 4-NN" % adj.sum(1).mean())

# ---------------- Part 1: inequality indices ----------------
def wgini(y, w):
    o = np.argsort(y); y, w = y[o], w[o]
    cw = np.cumsum(w) / w.sum(); cy = np.cumsum(y * w) / (y * w).sum()
    return 1 - np.sum((cy + np.r_[0, cy[:-1]]) * np.diff(np.r_[0, cw]))

def theil(y, w):
    s = w / w.sum(); mu = np.sum(s * y)
    return np.sum(s * (y / mu) * np.log(y / mu))

def sii_rii(y, w, rank_var):
    """Slope index of inequality: weighted regression of y on the population ridit rank (0 = least populous, 1 = most)."""
    o = np.argsort(rank_var); w_o = w[o]
    ridit = (np.cumsum(w_o) - 0.5 * w_o) / w_o.sum()
    X = np.column_stack([np.ones(len(y)), ridit]); Wm = np.diag(w_o)
    b = np.linalg.solve(X.T @ Wm @ X, X.T @ Wm @ y[o])
    mu = np.sum(w * y) / w.sum()
    return b[1], b[1] / mu, 2 * np.cov(y[o], ridit, aweights=w_o)[0, 1] / mu     # SII, RII, concentration index

rows = []
for yr in pv.columns:
    y, w = pv[yr].values, pop[yr].values
    sii, rii, ci = sii_rii(y, w, w)
    mu = np.average(y, weights=w)
    rows.append({"year": yr, "mean_w": mu, "sd": y.std(ddof=1), "range": y.max() - y.min(), "cv": y.std(ddof=1) / y.mean(),
                 "p90_p10": np.quantile(y, .9) / np.quantile(y, .1), "gini_w": wgini(y, w), "theil_w": theil(y, w),
                 "SII": sii, "RII": rii, "conc_index": ci})
ineq = pd.DataFrame(rows).round(4); ineq.to_csv(tabpath("ineq_indices_by_year.csv"), index=False, encoding="utf-8-sig")
print(ineq.set_index("year").loc[[1979, 1989, 1999, 2009, 2019, 2022], ["mean_w", "sd", "cv", "gini_w", "theil_w", "SII", "RII"]].round(3).to_string())

# ---------------- Part 2: spatial regressions ----------------
n = len(names)
def eig_real(W): return np.linalg.eigvals(W).real
def conc_ll(rho, y, X, W, eigs, kind):
    I = np.eye(len(y))
    A = I - rho * W
    Ay, AX = (A @ y, X) if kind == "lag" else (A @ y, A @ X)
    b = np.linalg.lstsq(AX, Ay, rcond=None)[0]; e = Ay - AX @ b; s2 = e @ e / len(y)
    return -len(y) / 2 * (np.log(2 * np.pi) + np.log(s2) + 1) + np.sum(np.log(np.abs(1 - rho * eigs))), b, e, s2

def fit_spatial(y, X, W, kind):
    eigs = eig_real(W)
    lo, hi = 1 / eigs.min() + 1e-3, 1 / eigs.max() - 1e-3
    r = minimize_scalar(lambda r: -conc_ll(r, y, X, W, eigs, kind)[0], bounds=(lo, hi), method="bounded")
    ll, b, e, s2 = conc_ll(r.x, y, X, W, eigs, kind)
    ll0 = conc_ll(0.0, y, X, W, eigs, kind)[0]
    # curvature-based s.e. of rho (profile likelihood; approximation, ignores covariance with beta and sigma^2)
    h = 1e-3
    f = lambda r_: conc_ll(r_, y, X, W, eigs, kind)[0]
    d2 = (f(r.x + h) - 2 * f(r.x) + f(r.x - h)) / h ** 2
    se = np.sqrt(-1 / d2) if d2 < 0 else np.nan
    lr = 2 * (ll - ll0)
    return {"rho": r.x, "se_rho": se, "lr": lr, "p_lr": stats.chi2.sf(max(lr, 0), 1), "beta": b, "ll": ll, "aic": -2 * ll + 2 * (X.shape[1] + 2)}

def lm_tests(y, X, W):
    b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b; s2 = e @ e / len(y)
    M = np.eye(len(y)) - X @ np.linalg.solve(X.T @ X, X.T)
    T = np.trace(W.T @ W + W @ W)
    nJ = ((W @ X @ b) @ M @ (W @ X @ b)) / s2 + T
    Ie, Il = e @ W @ e / s2, e @ W @ y / s2
    lm_err, lm_lag = Ie ** 2 / T, Il ** 2 / nJ
    rlm_err = (Ie - T / nJ * Il) ** 2 / (T - T ** 2 / nJ)
    rlm_lag = (Il - Ie) ** 2 / (nJ - T)
    sp = lambda x: stats.chi2.sf(x, 1)
    return {"LM_lag_p": sp(lm_lag), "LM_err_p": sp(lm_err), "RLM_lag_p": sp(rlm_lag), "RLM_err_p": sp(rlm_err)}

def moran(x, W):
    z = x - x.mean(); return len(x) * (z @ W @ z) / (z @ z) / W.sum()

def moran_p(x, W, B=4999):
    obs = moran(x, W); sims = np.array([moran(rng.permutation(x), W) for _ in range(B)])
    return obs, (1 + (sims >= obs).sum()) / (B + 1)

WINDOWS = [(1979, 1983), (1984, 1988), (1989, 1993), (1994, 1998), (1999, 2003), (2004, 2008), (2009, 2013), (2014, 2018), (2019, 2022)]
oral = d.pivot(index="county", columns="year", values="oral_asir").loc[names]
res = []
for (a, b_), (Wn, W) in [(w, ww) for w in WINDOWS for ww in (("queen", W_q), ("knn4", W_k))]:
    y = np.log(pv.loc[:, a:b_].mean(1).values)
    lp = np.log(pop.loc[:, a:b_].mean(1).values)
    lo = np.log(oral.loc[:, a:b_].mean(1).values)
    for spec, X in [("const", np.ones((n, 1))), ("ln pop", np.column_stack([np.ones(n), lp])),
                    ("ln pop + oral proxy", np.column_stack([np.ones(n), lp, lo]))]:
        if np.isnan(X).any(): continue
        ols_b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ ols_b
        mi, mp = moran_p(e, W, 1999)
        sar, sem = fit_spatial(y, X, W, "lag"), fit_spatial(y, X, W, "err")
        row = {"window": f"{a}-{b_}", "weights": Wn, "spec": spec, "moran_resid": mi, "moran_p": mp, "ols_r2": 1 - e @ e / ((y - y.mean()) @ (y - y.mean())),
               "sar_rho": sar["rho"], "sar_se": sar["se_rho"], "sar_p_lr": sar["p_lr"], "sar_aic": sar["aic"],
               "sem_lambda": sem["rho"], "sem_p_lr": sem["p_lr"], "sem_aic": sem["aic"],
               "ols_aic": n * (np.log(2 * np.pi) + np.log(e @ e / n) + 1) + 2 * (X.shape[1] + 1), **lm_tests(y, X, W)}
        for j, nm in enumerate(["const", "ln_pop", "ln_oral"][:X.shape[1]]):
            row[f"ols_b_{nm}"] = ols_b[j]; row[f"sar_b_{nm}"] = sar["beta"][j]
        res.append(row)
res = pd.DataFrame(res).round(4)
res.to_csv(tabpath("sar_models_by_window.csv"), index=False, encoding="utf-8-sig")
show = res[(res.weights == "queen") & (res.spec == "ln pop")][["window", "moran_resid", "moran_p", "sar_rho", "sar_p_lr", "sem_lambda", "sem_p_lr", "LM_lag_p", "LM_err_p", "ols_b_ln_pop", "sar_b_ln_pop"]]
print("\nSAR / SEM by window (queen weights, ln pop):\n", show.to_string(index=False))
print("\n4-NN weights (const):\n", res[(res.weights == "knn4") & (res.spec == "const")][["window", "moran_resid", "moran_p", "sar_rho", "sar_p_lr"]].to_string(index=False))
print("\nwith oral proxy (queen):\n", res[(res.weights == "queen") & (res.spec == "ln pop + oral proxy")][["window", "sar_rho", "sar_p_lr", "ols_b_ln_pop", "ols_b_ln_oral", "sar_b_ln_oral"]].to_string(index=False))

# ---------------- Part 3: spatial beta-convergence ----------------
y0 = np.log(pv.loc[:, 1979:1983].mean(1).values); y1 = np.log(pv.loc[:, 2018:2022].mean(1).values)
gr = (y1 - y0) / 39
conv = []
for Wn, W in (("queen", W_q), ("knn4", W_k)):
    for spec, X in [("beta only", np.column_stack([np.ones(n), y0])), ("beta + ln pop", np.column_stack([np.ones(n), y0, np.log(pop.loc[:, 1979:1983].mean(1).values)]))]:
        ols_b = np.linalg.lstsq(X, gr, rcond=None)[0]; sar = fit_spatial(gr, X, W, "lag")
        conv.append({"weights": Wn, "spec": spec, "ols_beta": ols_b[1], "sar_beta": sar["beta"][1], "sar_rho": sar["rho"], "sar_p_lr": sar["p_lr"],
                     "moran_growth": moran(gr, W), **lm_tests(gr, X, W)})
conv = pd.DataFrame(conv).round(4); conv.to_csv(tabpath("sar_spatial_convergence.csv"), index=False, encoding="utf-8-sig")
print("\nspatial convergence:\n", conv.to_string(index=False))

# ---------------- figures ----------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for k, lab in [("cv", "變異係數 CV（相對）"), ("gini_w", "人口加權 Gini（相對）"), ("theil_w", "Theil-T（相對）")]:
    ax[0].plot(ineq.year, ineq[k] / ineq[k].iloc[0], label=lab)
ax[0].plot(ineq.year, ineq.sd / ineq.sd.iloc[0], "k--", label="標準差 SD（絕對）"); ax[0].plot(ineq.year, ineq["range"] / ineq["range"].iloc[0], "k:", label="全距（絕對）")
ax[0].set(title="不平等指標（1979 = 1）：相對 vs 絕對", xlabel="年"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[1].plot(ineq.year, ineq.SII, label="SII（絕對，人口規模梯度）"); ax[1].axhline(0, c="k", lw=.6)
ax2 = ax[1].twinx(); ax2.plot(ineq.year, ineq.RII, c="tab:red", label="RII（相對）"); ax2.set_ylabel("RII")
ax[1].set(title="依縣市人口規模的斜率不平等指標", xlabel="年", ylabel="SII（每10萬）"); ax[1].grid(alpha=.3)
fig.savefig(figpath("ineq_trends.png"), dpi=200, bbox_inches="tight")

sub = res[(res.weights == "queen") & (res.spec == "const")]
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(sub))
ax.errorbar(x, sub.sar_rho, yerr=1.96 * sub.sar_se, fmt="o-", capsize=3, label="SAR ρ（±1.96se，近似）")
ax.plot(x, sub.moran_resid, "s--", label="殘差 Moran's I"); ax.axhline(0, c="k", lw=.6)
ax.set_xticks(x); ax.set_xticklabels(sub.window, rotation=30); ax.grid(alpha=.3); ax.legend()
ax.set(title="空間依賴隨時間消退：SAR ρ（僅截距）與 Moran's I", ylabel="係數")
fig.savefig(figpath("sar_rho_by_window.png"), dpi=200, bbox_inches="tight")
