"""Double/debiased machine learning (DML) for difference-in-differences, with cross-fitting.

Units    : county x cancer-site series (both sexes, or the relevant sex for sex-specific sites), 21 counties x ~30 sites.
Treatment: county in {Taipei, New Taipei} (epicentre of the 2021 outbreak). Assignment is at the COUNTY level.
Outcome  : change in ln ASIR between the baseline window 2017-2019 and the post year(s) (2020 / 2021 / 2022 / 2021-22).
Covariates X (all pre-treatment): site, sex stratum, baseline level, pre-trend (2017-19 vs 2014-16), pre-period volatility,
          log case count, plus county-level ln population.

Estimators (Chernozhukov et al. 2018 Neyman-orthogonal scores; Sant'Anna & Zhao 2020 doubly robust DiD):
  OR-DML : ATT = mean_{treated}( dY - m0(X) ),  m0 fitted on controls by ML, cross-fitted by COUNTY folds.
  DR-DML : ATT = E[ (D/p - pi(1-D)/(p(1-pi))) (dY - m0(X)) ]  with propensity pi(X) from L2 logistic regression (clipped).
Learners : gradient boosting / random forest / lasso (reported side by side).
Inference : the number of treated COUNTIES is two, so analytic / cluster-robust standard errors are unreliable.  We use
           exact randomisation inference: re-run the whole cross-fitted estimator for each of the C(21,2)=210 placebo
           pairs of counties and compare the observed ATT with that distribution.
Checks   : placebo-in-time (pretend the outbreak was in 2017 / 2018), heterogeneity by site group (screened / unscreened / lung)."""
import itertools
import os
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # avoid thread oversubscription inside joblib workers
    os.environ[_v] = "1"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.model_selection import GroupKFold

from he_common import RAW_CSV, figpath, load_panel, setup_plot, tabpath

warnings.filterwarnings("ignore")
setup_plot()
rng = np.random.default_rng(42)
TREATED = ["台北市", "新北市"]
MIN_CASES = 15          # mean annual cases over 2015-19 for a county-site series to enter
SCREENED = {"結直腸", "口腔、口咽及下咽", "女性乳房", "子宮頸"}

# ------------------------------- build county x site panel ---------------------------------------------------
raw = pd.read_csv(RAW_CSV)
raw.columns = ["year", "sex", "county", "site", "asir", "n", "mean_age", "median_age", "crude"]
raw["n"] = pd.to_numeric(raw["n"].astype(str).str.replace(",", "").str.strip())
EXC = ["全癌症", "不明原發部位", "其他分界不明的部位", "消化器官其他分界不明部位", "其他男性生殖器官", "其他女性生殖器官", "其他神經系統",
       "其他內分泌腺", "子宮", "男性乳房", "胸腺、心臟與中隔", "胸膜", "後腹膜腔及腹膜", "眼及淚腺", "睪丸", "小腸", "主唾液腺"]
raw = raw[(raw.county != "全國") & (raw.county != "連江縣") & (~raw.site.isin(EXC))]
# keep the "all" stratum when it exists, else the single-sex stratum
has_all = raw[raw.sex == "全"].site.unique()
raw = raw[((raw.sex == "全") & raw.site.isin(has_all)) | (~raw.site.isin(has_all) & raw.sex.isin(["男", "女"]))]
raw["key"] = raw.county + "|" + raw.site + "|" + raw.sex
asir = raw.pivot(index="key", columns="year", values="asir")
cases = raw.pivot(index="key", columns="year", values="n")
keep = (cases.loc[:, 2015:2019].mean(1) >= MIN_CASES) & (asir.loc[:, 2012:2022] > 0).all(1)
asir, cases = asir[keep], cases[keep]
lnA = np.log(asir)
meta = pd.DataFrame({"key": asir.index})
meta[["county", "site", "sex"]] = meta.key.str.split("|", expand=True)
meta["treated"] = meta.county.isin(TREATED).astype(int)
meta["screened"] = meta.site.isin(SCREENED).astype(int)
meta["lung"] = (meta.site == "肺、支氣管及氣管").astype(int)
pop = load_panel().query("year == 2019").set_index("county")["pop"]
meta["ln_pop"] = np.log(meta.county.map(pop)).values
print(f"{len(meta)} county-site series | counties {meta.county.nunique()} | sites {meta.site.nunique()} | treated series {meta.treated.sum()}")

def features(base_years, pre_years):
    """covariates from windows strictly before the post period (base window and an earlier pre-trend window)."""
    lvl = lnA.loc[:, base_years].mean(1).values
    trend = lvl - lnA.loc[:, pre_years].mean(1).values
    vol = lnA.loc[:, range(2010, base_years[0])].diff(axis=1).std(1).values
    X = pd.DataFrame({"base_level": lvl, "pre_trend": trend, "pre_vol": vol, "ln_cases": np.log(cases.loc[:, base_years].mean(1).values),
                      "ln_pop": meta.ln_pop.values, "screened": meta.screened.values})
    return pd.concat([X, pd.get_dummies(meta.site, prefix="s").astype(float), pd.get_dummies(meta.sex, prefix="x").astype(float)], axis=1)

# ------------------------------- estimators --------------------------------------------------------------------
def make_learner(name):
    if name == "GBM": return lambda: GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=.05, subsample=.8, random_state=0)
    if name == "RF": return lambda: RandomForestRegressor(n_estimators=100, min_samples_leaf=8, max_features=.5, random_state=0, n_jobs=1)
    if name == "Lasso": return lambda: LassoCV(cv=5, random_state=0, max_iter=5000)
    raise ValueError(name)

def att(dy, D, X, county, learner, n_splits=5, dr=False):
    """Cross-fitted OR / DR estimate of the ATT; folds grouped by county so a county's units never train and test together."""
    n = len(dy); m0 = np.zeros(n); pi = np.zeros(n)
    folds = GroupKFold(n_splits=n_splits)
    Xs = (X - X.mean()) / (X.std() + 1e-9)
    for tr, te in folds.split(Xs, dy, groups=county):
        ctrl = tr[D[tr] == 0]
        m0[te] = make_learner(learner)().fit(Xs.iloc[ctrl], dy[ctrl]).predict(Xs.iloc[te])
        if dr:
            cols = [c for c in X.columns if c in ("base_level", "pre_trend", "pre_vol", "ln_cases", "screened")]
            if len(set(D[tr])) < 2:        # a placebo split can leave no treated county in the training folds
                pi[te] = np.clip(D[tr].mean(), 0.01, 0.99)
            else:
                lr = LogisticRegression(C=0.5, max_iter=1000).fit(Xs.iloc[tr][cols], D[tr])
                pi[te] = np.clip(lr.predict_proba(Xs.iloc[te][cols])[:, 1], 0.01, 0.99)
    res = dy - m0
    if not dr: return res[D == 1].mean()
    p = D.mean(); w = D / p - pi * (1 - D) / (p * (1 - pi))
    return np.mean(w * res)

def ri_p(dy, X, county, learner, obs, dr=False, treated_set=TREATED):
    counties = np.array(sorted(set(county)))
    dist = np.array(Parallel(n_jobs=4)(
        delayed(att)(dy, np.isin(county, pair).astype(int), X, county, learner, dr=dr)
        for pair in itertools.combinations(counties, len(treated_set))))
    return (np.sum(np.abs(dist) >= abs(obs) - 1e-12)) / len(dist), dist

BASE, PRE = list(range(2017, 2020)), list(range(2014, 2017))
X = features(BASE, PRE)
county = meta.county.values; D = meta.treated.values
OUTCOMES = {"2020": [2020], "2021": [2021], "2022": [2022], "2021-22": [2021, 2022]}
rows, ri_store = [], {}
for learner in ["GBM", "RF", "Lasso"]:
    for oname, yrs in OUTCOMES.items():
        dy = (lnA.loc[:, yrs].mean(1).values - lnA.loc[:, BASE].mean(1).values)
        for dr in (False, True):
            a = att(dy, D, X, county, learner, dr=dr)
            # randomisation inference (210 placebo pairs x cross-fitting) only for the headline outcomes and the two fast learners
            do_ri = oname in ("2021", "2021-22") and learner in ("GBM", "Lasso")
            p, dist = ri_p(dy, X, county, learner, a, dr=dr) if do_ri else (np.nan, None)
            rows.append({"learner": learner, "estimator": "DR-DML" if dr else "OR-DML", "post": oname, "att_logpts": a, "att_pct": (np.exp(a) - 1) * 100,
                         "ri_p": p, "placebo_sd": np.std(dist) if dist is not None else np.nan, "n_series": len(dy)})
            if dist is not None and learner == "GBM": ri_store[(oname, dr)] = (a, dist)
            print(f"{learner:5s} {('DR' if dr else 'OR'):2s} post={oname:8s}  ATT = {a*100:6.2f} log pts ({(np.exp(a)-1)*100:5.1f}%)  RI p = {p:.3f}")
res = pd.DataFrame(rows).round(4); res.to_csv(tabpath("dml_did_att.csv"), index=False, encoding="utf-8-sig")

# naive benchmarks: unadjusted DID on the same series (no covariates), and TWFE-like mean difference
dy21 = lnA.loc[:, [2021, 2022]].mean(1).values - lnA.loc[:, BASE].mean(1).values
naive = dy21[D == 1].mean() - dy21[D == 0].mean()
print(f"\nnaive DID (no covariates), 2021-22: {naive*100:.2f} log pts")
pd.DataFrame([{"estimator": "naive DID (series means)", "post": "2021-22", "att_logpts": naive, "att_pct": (np.exp(naive) - 1) * 100}]).round(4).to_csv(
    tabpath("dml_did_naive.csv"), index=False)

# ------------------------------- heterogeneity by site group (DR scores from the GBM OR model) -----------------
dy = dy21; Xs = (X - X.mean()) / (X.std() + 1e-9)
m0 = np.zeros(len(dy))
for tr, te in GroupKFold(5).split(Xs, dy, groups=county):
    ctrl = tr[D[tr] == 0]; m0[te] = make_learner("GBM")().fit(Xs.iloc[ctrl], dy[ctrl]).predict(Xs.iloc[te])
res_i = dy - m0
grp = np.where(meta.lung == 1, "肺癌", np.where(meta.screened == 1, "篩檢導向癌別", "其他癌別"))
het = []
counties_all = np.array(sorted(set(county)))
GROUPS = ["肺癌", "篩檢導向癌別", "其他癌別"]
def _pair(pair):
    """Lasso OR-DML residual means by site group when `pair` of counties is (placebo-)treated."""
    Dp = np.isin(county, pair).astype(int); m0p = np.zeros(len(dy))
    for tr, te in GroupKFold(5).split(Xs, dy, groups=county):
        ctrl = tr[Dp[tr] == 0]; m0p[te] = make_learner("Lasso")().fit(Xs.iloc[ctrl], dy[ctrl]).predict(Xs.iloc[te])
    r = dy - m0p
    return [r[(grp == g) & (Dp == 1)].mean() for g in GROUPS]
dist_all = np.array(Parallel(n_jobs=4)(delayed(_pair)(p) for p in itertools.combinations(counties_all, 2)))
obs_g = _pair(tuple(TREATED))
for j, g in enumerate(GROUPS):
    sel = (grp == g) & (D == 1)
    het.append({"group": g, "n_treated_series": int(sel.sum()), "att_logpts": obs_g[j], "att_pct": (np.exp(obs_g[j]) - 1) * 100,
                "ri_p_lasso_placebo": np.mean(np.abs(dist_all[:, j]) >= abs(obs_g[j]) - 1e-12), "placebo_sd": np.std(dist_all[:, j])})
het = pd.DataFrame(het).round(4); het.to_csv(tabpath("dml_did_heterogeneity.csv"), index=False, encoding="utf-8-sig")
print("\nheterogeneity (GBM OR-DML, 2021-22):\n", het.to_string(index=False))
bysite = pd.DataFrame({"site": meta.site, "treated": D, "res": res_i})
bysite = bysite[bysite.treated == 1].groupby("site").res.mean().sort_values()
bysite.round(4).to_csv(tabpath("dml_did_by_site.csv"), encoding="utf-8-sig")

# ------------------------------- placebo-in-time ------------------------------------------------------------------
plc = []
for fake in (2017, 2018):
    base_p, pre_p = list(range(fake - 3, fake)), list(range(fake - 6, fake - 3))
    Xp = features(base_p, pre_p)
    dyp = lnA.loc[:, fake].values - lnA.loc[:, base_p].mean(1).values
    a = att(dyp, D, Xp, county, "GBM")
    p, _ = ri_p(dyp, Xp, county, "GBM", a)
    plc.append({"fake_post_year": fake, "att_logpts": a, "att_pct": (np.exp(a) - 1) * 100, "ri_p": p})
    print(f"placebo-in-time {fake}: ATT {a*100:.2f} log pts, RI p = {p:.2f}")
pd.DataFrame(plc).round(4).to_csv(tabpath("dml_did_placebo_time.csv"), index=False)

# ------------------------------- figures --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, oname in zip(axes, ["2021", "2021-22"]):
    a, dist = ri_store[(oname, False)]
    ax.hist(dist * 100, bins=30, color="grey", alpha=.6); ax.axvline(a * 100, c="r", lw=2, label=f"雙北 ATT = {a*100:.1f}")
    ax.legend(); ax.set(title=f"隨機化推論（OR-DML, GBM）：疫後 {oname}", xlabel="ATT (log points ×100)", ylabel="安慰劑指派數"); ax.grid(alpha=.3)
fig.savefig(figpath("dml_ri_distribution.png"), dpi=200, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(bysite.index, bysite.values * 100, color=np.where(bysite.values > 0, "tab:red", "tab:blue"))
ax.axvline(0, c="k", lw=.7); ax.grid(alpha=.3)
ax.set(title="雙北各癌別 ATT（2021–22，OR-DML；僅描述，單一癌別估計雜訊大）", xlabel="log points ×100")
fig.savefig(figpath("dml_did_by_site.png"), dpi=200, bbox_inches="tight")
