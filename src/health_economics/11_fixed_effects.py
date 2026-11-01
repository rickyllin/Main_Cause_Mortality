"""Fixed-effects panel models for the long-run trend of lung-cancer incidence (21 counties x 1979-2022).

Outcome  y_ct = ln ASIR_ct  (coefficient x 100 ~ percent change).
 M1  pooled OLS                     y = a + b t
 M2  county FE + linear trend        y = a_c + b t
 M3  county FE + quadratic trend     y = a_c + b1 t + b2 t^2
 M4  county FE + county-specific linear trends (heterogeneous growth); F-test of equal slopes
 M5  two-way FE (county + year)      -> common year profile, variance decomposition
 M6  two-way FE + ln population (+ oral-cancer risk proxy)  -> within-county association (descriptive)
Standard errors are clustered by county (21 clusters, t with G-1 df). WLS (weights = case counts) as robustness.
Also: sigma- and beta-convergence of county rates."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from he_common import figpath, load_panel, setup_plot, tabpath

setup_plot()
d = load_panel().copy()
d["t"] = d.year - 1979
d["t2"] = d.t ** 2 / 100
d["ln_oral"] = np.log(d.oral_asir)
rows = []

def fit(formula, name, data=d, weights=None, keep=None):
    m = (smf.wls(formula, data, weights=weights) if weights is not None else smf.ols(formula, data)).fit(
        cov_type="cluster", cov_kwds={"groups": data.county.astype("category").cat.codes, "use_correction": True}, use_t=True)
    ci = m.conf_int()
    for term in (keep or [t for t in m.params.index if not t.startswith(("C(", "Intercept"))]):
        rows.append({"model": name, "term": term, "coef": m.params[term], "se": m.bse[term], "p": m.pvalues[term],
                     "ci_lo": ci.loc[term, 0], "ci_hi": ci.loc[term, 1], "n_obs": int(m.nobs), "r2": m.rsquared})
    return m

m1 = fit("ln_asir ~ t", "M1 pooled OLS", keep=["t"])
m2 = fit("ln_asir ~ t + C(county)", "M2 county FE + trend")
m2w = fit("ln_asir ~ t + C(county)", "M2w county FE + trend (WLS, weight=cases)", weights=d.n)
m3 = fit("ln_asir ~ t + t2 + C(county)", "M3 county FE + quadratic")
b1, b2 = m3.params["t"], m3.params["t2"]
turning = -b1 / (2 * b2 / 100) + 1979          # t* = -b1 / (2*b2/100)
print(f"M1 pooled trend {m1.params['t']*100:.2f}%/yr | M2 FE trend {m2.params['t']*100:.2f}%/yr "
      f"[{m2.conf_int().loc['t',0]*100:.2f},{m2.conf_int().loc['t',1]*100:.2f}] | WLS {m2w.params['t']*100:.2f}%/yr")
print(f"M3 quadratic: b1={b1:.4f}, b2/100={b2/100:.6f} -> turning point of growth rate ~ {turning:.0f}")

# M4: county-specific slopes (separate OLS, HAC SE) + F-test for equal slopes
slopes = []
for c, g in d.groupby("county"):
    r = sm.OLS(g.ln_asir, sm.add_constant(g.t)).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    slopes.append({"county": c, "slope": r.params["t"], "se": r.bse["t"], "alpha": r.params["const"], "r2": r.rsquared})
sl = pd.DataFrame(slopes).sort_values("slope")
sl["annual_growth_pct"] = (np.exp(sl.slope) - 1) * 100
sl["lo"] = (np.exp(sl.slope - 1.96 * sl.se) - 1) * 100; sl["hi"] = (np.exp(sl.slope + 1.96 * sl.se) - 1) * 100
sl.round(4).to_csv(tabpath("fe_county_specific_slopes.csv"), index=False, encoding="utf-8-sig")
f_het = anova_lm(smf.ols("ln_asir ~ t + C(county)", d).fit(), smf.ols("ln_asir ~ C(county) * t", d).fit())
rows.append({"model": "M4 test: equal county slopes (F)", "term": "F", "coef": f_het["F"].iloc[1], "p": f_het["Pr(>F)"].iloc[1],
             "n_obs": len(d)})
print(f"M4 equal-slope F = {f_het['F'].iloc[1]:.2f}, p = {f_het['Pr(>F)'].iloc[1]:.2g}")
print(sl[["county", "annual_growth_pct", "lo", "hi"]].round(2).to_string(index=False))

# M5: two-way FE, year effects relative to 1979; variance decomposition
m5 = smf.ols("ln_asir ~ C(county) + C(year)", d).fit(cov_type="cluster", use_t=True,
        cov_kwds={"groups": d.county.astype("category").cat.codes, "use_correction": True})
yr = pd.DataFrame({"year": range(1980, 2023)})
yr["effect"] = [m5.params[f"C(year)[T.{y}]"] for y in yr.year]
yr["se"] = [m5.bse[f"C(year)[T.{y}]"] for y in yr.year]
yr = pd.concat([pd.DataFrame({"year": [1979], "effect": [0.0], "se": [0.0]}), yr])
yr["pct_vs_1979"] = (np.exp(yr.effect) - 1) * 100
yr.round(4).to_csv(tabpath("fe_year_effects.csv"), index=False, encoding="utf-8-sig")
r2_c = smf.ols("ln_asir ~ C(county)", d).fit().rsquared
r2_y = smf.ols("ln_asir ~ C(year)", d).fit().rsquared
r2_cy = m5.rsquared
vd = pd.DataFrame({"component": ["county FE only", "year FE only", "county + year FE", "unexplained"],
                   "share_of_variance": [r2_c, r2_y, r2_cy, 1 - r2_cy]})
vd.round(4).to_csv(tabpath("fe_variance_decomposition.csv"), index=False, encoding="utf-8-sig")
print(vd.round(3).to_string(index=False))
ar1 = np.corrcoef(m5.resid.values[:-1], m5.resid.values[1:])[0, 1]
print(f"TWFE residual lag-1 correlation (pooled) = {ar1:.2f}")

# M6: two-way FE + covariates (within-county association)
dd = d.dropna(subset=["ln_oral"])
m6 = fit("ln_asir ~ ln_pop + C(county) + C(year)", "M6a TWFE + ln pop", data=dd, keep=["ln_pop"])
m6b = fit("ln_asir ~ ln_pop + ln_oral + C(county) + C(year)", "M6b TWFE + ln pop + ln oral-cancer ASIR", data=dd, keep=["ln_pop", "ln_oral"])
fit("ln_asir ~ ln_pop + ln_oral + C(county) + C(year)", "M6c TWFE + covariates (WLS)", data=dd, weights=dd.n, keep=["ln_pop", "ln_oral"])
res = pd.DataFrame(rows); res.round(5).to_csv(tabpath("fe_models.csv"), index=False, encoding="utf-8-sig")
print(res[res.model.str.startswith(("M6"))][["model", "term", "coef", "se", "p"]].round(3).to_string(index=False))

# sigma / beta convergence
w = d.pivot(index="county", columns="year", values="ln_asir")
sigma = w.std()
first, last = w.loc[:, 1979:1983].mean(1), w.loc[:, 2018:2022].mean(1)
g = (last - first) / 39
bc = smf.ols("g ~ first", pd.DataFrame({"g": g, "first": first})).fit(cov_type="HC3")
print(f"beta-convergence: slope on initial ln ASIR = {bc.params['first']:.4f} (se {bc.bse['first']:.4f}, p={bc.pvalues['first']:.3g}), R2={bc.rsquared:.2f}")
print(f"sigma: SD of ln ASIR 1979 = {sigma[1979]:.2f}, 1989 = {sigma[1989]:.2f}, 2009 = {sigma[2009]:.2f}, 2022 = {sigma[2022]:.2f}")
pd.DataFrame({"year": sigma.index, "sd_ln_asir": sigma.values}).round(4).to_csv(tabpath("fe_sigma_convergence.csv"), index=False)
pd.DataFrame([{"slope": bc.params["first"], "se": bc.bse["first"], "p": bc.pvalues["first"], "r2": bc.rsquared,
               "half_life_years": np.log(2) / -np.log(1 + 39 * bc.params["first"]) * 39 if bc.params["first"] < 0 else np.nan}]
             ).round(4).to_csv(tabpath("fe_beta_convergence.csv"), index=False)

# ---- figures --------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.fill_between(yr.year, (np.exp(yr.effect - 1.96 * yr.se) - 1) * 100, (np.exp(yr.effect + 1.96 * yr.se) - 1) * 100, alpha=.25)
ax.plot(yr.year, yr.pct_vs_1979, lw=2); ax.axvline(2021, ls="--", c="grey"); ax.grid(alpha=.3)
ax.set(title="雙向固定效果模型：共同時間效應（相對 1979 年，95% 可信帶，縣市群聚標準誤）", xlabel="診斷年", ylabel="相對 1979 年之發生率變化 (%)")
fig.savefig(figpath("fe_year_effects.png"), dpi=200, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(8, 8)); y = np.arange(len(sl))
ax.errorbar(sl.annual_growth_pct, y, xerr=[sl.annual_growth_pct - sl.lo, sl.hi - sl.annual_growth_pct], fmt="o", capsize=3)
ax.axvline((np.exp(m2.params["t"]) - 1) * 100, ls="--", c="r", label="共同斜率（M2）"); ax.legend()
ax.set_yticks(y); ax.set_yticklabels(sl.county); ax.grid(alpha=.3)
ax.set(title="各縣市年平均成長率（縣市別線性趨勢，95%CI）", xlabel="年平均成長率 (%)")
fig.savefig(figpath("fe_county_slopes.png"), dpi=200, bbox_inches="tight")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot(sigma.index, sigma.values, lw=2); ax[0].set(title="σ 收斂：縣市間 ln ASIR 標準差", xlabel="年"); ax[0].grid(alpha=.3)
ax[1].scatter(first, g * 100)
for c in first.index: ax[1].annotate(c, (first[c], g[c] * 100), xytext=(3, 3), textcoords="offset points", fontsize=8)
xs = np.linspace(first.min(), first.max(), 50); ax[1].plot(xs, (bc.params["Intercept"] + bc.params["first"] * xs) * 100, c="r")
ax[1].set(title="β 收斂：期初水準 vs 年成長", xlabel="期初 ln ASIR（1979–83 平均）", ylabel="年成長 (log points ×100)"); ax[1].grid(alpha=.3)
fig.savefig(figpath("fe_beta_convergence.png"), dpi=200, bbox_inches="tight")
