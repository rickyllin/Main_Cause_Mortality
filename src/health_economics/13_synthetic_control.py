"""Synthetic control and augmented (ridge) synthetic control for the 2021 outbreak in Taipei + New Taipei.

Treated unit : population-weighted lung-cancer ASIR of Taipei + New Taipei (ln scale).
Donor pool   : the other 19 counties (variants: drop Keelung/Taoyuan = possible spill-over; drop the volatile islands).
Pre-period   : 2005-2019, post-period 2020-2022.
SC   : non-negative weights summing to one, minimising the pre-period RMSPE.
ASCM : SC + ridge-regression bias correction (Ben-Michael, Feller & Rothstein 2021), penalty chosen by leave-one-donor-out CV.
Inference: in-space placebo (every donor treated as if it were the intervention unit), post/pre RMSPE ratio and gap rank."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from he_common import figpath, load_panel, setup_plot, tabpath

setup_plot()
TREATED = ["台北市", "新北市"]
PRE, POST = list(range(2005, 2020)), [2020, 2021, 2022]
d = load_panel()
ln = d.pivot(index="county", columns="year", values="ln_asir")
# population-weighted treated series (average ASIR, then log)
tr = d[d.county.isin(TREATED)].groupby("year").apply(lambda g: np.average(g.asir, weights=g["pop"]))
y1 = np.log(tr)

def sc_weights(y_pre, Y0_pre):
    J = Y0_pre.shape[0]
    obj = lambda w: np.sum((y_pre - w @ Y0_pre) ** 2)
    r = minimize(obj, np.full(J, 1 / J), method="SLSQP", bounds=[(0, 1)] * J,
                 constraints={"type": "eq", "fun": lambda w: w.sum() - 1}, options={"maxiter": 500, "ftol": 1e-12})
    return r.x

def ridge_fit(Xp, Yp, lam):
    """Ridge: Yp (donors x post-years) on Xp (donors x pre-years), centred. Returns predict(x)."""
    mu_x, mu_y = Xp.mean(0), Yp.mean(0)
    A = Xp - mu_x
    B = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Yp - mu_y))
    return lambda x: mu_y + (x - mu_x) @ B

def choose_lambda(Xp, Yp):
    grid = np.logspace(-3, 2, 20); best = (np.inf, None)
    for lam in grid:
        err = 0
        for i in range(len(Xp)):
            tr_i = np.delete(np.arange(len(Xp)), i)
            pred = ridge_fit(Xp[tr_i], Yp[tr_i], lam)(Xp[i]); err += np.sum((Yp[i] - pred) ** 2)
        if err < best[0]: best = (err, lam)
    return best[1]

def estimate(y1_pre, y1_post, donors, ln_df, pre=PRE, post=POST):
    Y0_pre, Y0_post = ln_df.loc[donors, pre].values, ln_df.loc[donors, post].values
    w = sc_weights(y1_pre, Y0_pre)
    sc_pre, sc_post = w @ Y0_pre, w @ Y0_post
    lam = choose_lambda(Y0_pre, Y0_post)
    m = ridge_fit(Y0_pre, Y0_post, lam)
    asc_post = sc_post + (m(y1_pre) - w @ np.array([m(x) for x in Y0_pre]))
    # pre-period fit of ASCM equals SC's (the correction only shifts post-period predictions)
    return {"w": w, "sc_pre": sc_pre, "sc_post": sc_post, "asc_post": asc_post, "lam": lam,
            "gap_pre": y1_pre - sc_pre, "gap_sc": y1_post - sc_post, "gap_asc": y1_post - asc_post}

def rmspe(x): return np.sqrt(np.mean(np.square(x)))

results = {}
VARIANTS = {
    "all19": [c for c in ln.index if c not in TREATED],
    "no_spillover": [c for c in ln.index if c not in TREATED + ["基隆市", "桃園市"]],
    "no_islands": [c for c in ln.index if c not in TREATED + ["金門縣", "澎湖縣"]],
}
rows, path_rows, w_rows = [], [], []
for vname, donors in VARIANTS.items():
    e = estimate(y1.loc[PRE].values, y1.loc[POST].values, donors, ln)
    # placebo-in-space: each donor as treated, remaining donors (excluding the real treated) as pool
    plac = []
    for dn in donors:
        pool = [c for c in donors if c != dn]
        pe = estimate(ln.loc[dn, PRE].values, ln.loc[dn, POST].values, pool, ln)
        plac.append({"unit": dn, "pre_rmspe": rmspe(pe["gap_pre"]), "avg_gap_sc": pe["gap_sc"].mean(), "avg_gap_asc": pe["gap_asc"].mean(),
                     "ratio_sc": rmspe(pe["gap_sc"]) / rmspe(pe["gap_pre"])})
    pl = pd.DataFrame(plac)
    obs = {"avg_gap_sc": e["gap_sc"].mean(), "avg_gap_asc": e["gap_asc"].mean(), "pre_rmspe": rmspe(e["gap_pre"]),
           "ratio_sc": rmspe(e["gap_sc"]) / rmspe(e["gap_pre"])}
    # inference on units with decent pre-fit (pre RMSPE <= 2x treated's) to avoid noise-driven placebos; also unrestricted
    good = pl[pl.pre_rmspe <= 2 * obs["pre_rmspe"] + 1e-9]
    p_sc = (np.sum(np.abs(good.avg_gap_sc) >= abs(obs["avg_gap_sc"])) + 1) / (len(good) + 1)
    p_asc = (np.sum(np.abs(good.avg_gap_asc) >= abs(obs["avg_gap_asc"])) + 1) / (len(good) + 1)
    p_ratio = (np.sum(good.ratio_sc >= obs["ratio_sc"]) + 1) / (len(good) + 1)
    p_all = (np.sum(np.abs(pl.avg_gap_sc) >= abs(obs["avg_gap_sc"])) + 1) / (len(pl) + 1)
    rows.append({"variant": vname, "pre_rmspe": obs["pre_rmspe"], "avg_post_gap_SC_pct": (np.exp(obs["avg_gap_sc"]) - 1) * 100,
                 "avg_post_gap_ASCM_pct": (np.exp(obs["avg_gap_asc"]) - 1) * 100, "p_SC_gap": p_sc, "p_ASCM_gap": p_asc, "p_SC_gap_all_placebos": p_all,
                 "p_rmspe_ratio": p_ratio, "n_placebo_used": len(good), "n_placebo_all": len(pl), "ridge_lambda": e["lam"]})
    for t, g1, g2 in zip(POST, e["gap_sc"], e["gap_asc"]):
        path_rows.append({"variant": vname, "year": t, "gap_sc": g1, "gap_asc": g2})
    for c, wv in zip(donors, e["w"]):
        if wv > 1e-3: w_rows.append({"variant": vname, "donor": c, "weight": wv})
    results[vname] = (e, pl, obs)
    print(f"\n[{vname}] pre-RMSPE {obs['pre_rmspe']:.3f} | avg post gap SC {rows[-1]['avg_post_gap_SC_pct']:.1f}% (p={p_sc:.2f}), "
          f"ASCM {rows[-1]['avg_post_gap_ASCM_pct']:.1f}% (p={p_asc:.2f}) | RMSPE-ratio p={p_ratio:.2f} | placebo used {len(good)}/{len(pl)}")
    print("  weights:", {c: round(w, 2) for c, w in zip(donors, e["w"]) if w > .01})
summ = pd.DataFrame(rows).round(4); summ.to_csv(tabpath("sc_summary.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(path_rows).round(4).to_csv(tabpath("sc_gap_paths.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(w_rows).round(4).to_csv(tabpath("sc_donor_weights.csv"), index=False, encoding="utf-8-sig")

# ---- figures (main variant) ------------------------------------------------------------------------------
e, pl, obs = results["all19"]
yrs_all = PRE + POST
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(yrs_all, np.exp(y1.loc[yrs_all]), "o-", c="tab:red", label="雙北（實際）")
ax.plot(yrs_all, np.exp(np.concatenate([e["sc_pre"], e["sc_post"]])), "s--", c="tab:blue", label="合成雙北（SC）")
ax.plot(POST, np.exp(e["asc_post"]), "^--", c="tab:green", label="ASCM 反事實")
ax.axvline(2020.5, ls=":", c="grey"); ax.grid(alpha=.3); ax.legend()
ax.set(title="合成控制：雙北肺癌發生率 vs. 合成對照", xlabel="年", ylabel="年齡標準化發生率（每10萬）")
fig.savefig(figpath("sc_fit.png"), dpi=200, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(10, 5.5))
for dn in VARIANTS["all19"]:
    pool = [c for c in VARIANTS["all19"] if c != dn]
    pe = estimate(ln.loc[dn, PRE].values, ln.loc[dn, POST].values, pool, ln)
    ax.plot(yrs_all, np.concatenate([pe["gap_pre"], pe["gap_sc"]]) * 100, c="grey", alpha=.4, lw=1)
ax.plot(yrs_all, np.concatenate([e["gap_pre"], e["gap_sc"]]) * 100, c="tab:red", lw=2.5, label="雙北")
ax.axhline(0, c="k", lw=.7); ax.axvline(2020.5, ls=":", c="grey"); ax.legend(); ax.grid(alpha=.3)
ax.set(title="安慰劑檢定（空間）：實際 − 合成之差距（log points ×100）；灰線為各對照縣市", xlabel="年", ylabel="差距")
fig.savefig(figpath("sc_placebo_gaps.png"), dpi=200, bbox_inches="tight")
