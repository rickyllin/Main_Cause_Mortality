"""Difference-in-differences around the 2021 COVID-19 outbreak in Taiwan.

Part A (geographic): outcome = ln ASIR of lung cancer, county x year (2009-2022).
    treated = Taipei + New Taipei (epicentre of the May-2021 local outbreak); robust variant adds Keelung + Taoyuan.
    TWFE: y_ct = a_c + g_t + sum_k b_k * Treat_c * 1[t=k] + e_ct,  event-study with base year 2019, 2020 shown separately.
    Inference: only 2-4 treated units, so cluster-robust SEs are unreliable -> exact randomisation inference over every possible
    assignment of the same number of treated counties (C(21,2)=210; C(21,4)=5985).
Part B (cancer site, national): screening-sensitive sites (colorectal, oral, female breast, cervix; organised national screening)
    vs. unscreened sites (stomach, liver, pancreas, oesophagus, bladder, NHL, leukaemia, brain, prostate); lung shown separately.
Treatment timing: 2021 = year of the outbreak/level-3 alert (May-July), 2022 = Omicron wave; 2020 = placebo-ish early year."""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from he_common import NATIONAL, figpath, load_panel, setup_plot, tabpath

setup_plot()
rng = np.random.default_rng(42)
BASE = 2019
YEARS = list(range(2009, 2023))
EVENT_YEARS = [y for y in range(2012, 2023) if y != BASE]


# ---------------- TWFE on a balanced unit x year matrix ----------------
def twoway_demean(M):
    return M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()


def twfe(Y, treat, years_used, ev_years, w=None, group_trend=False):
    """Y: units x years (ln outcome); treat: bool per unit. Returns {year: beta} for Treat x 1[year] (base excluded)."""
    yrs = np.array(years_used)
    Ydm = twoway_demean(Y).ravel()
    X = []
    for k in ev_years:
        D = np.outer(treat.astype(float), (yrs == k).astype(float)); X.append(twoway_demean(D).ravel())
    if group_trend:                                 # treated-specific linear trend
        X.append(twoway_demean(np.outer(treat.astype(float), yrs - yrs.mean())).ravel())
    X = np.column_stack(X)
    b = np.linalg.lstsq(X, Ydm, rcond=None)[0]
    return dict(zip(list(ev_years) + (["trend"] if group_trend else []), b))


def basic_did(Y, treat, years_used, post=(2021, 2022), early=(2020,), drop=()):
    """Treat x post(>=2021) [and Treat x 2020] with pre = years <= 2019; returns (b_post, b_2020)."""
    yrs = np.array(years_used); keep = np.array([y not in drop for y in yrs])
    Yk = Y[:, keep]; yk = yrs[keep]
    dpost = np.outer(treat.astype(float), np.isin(yk, post).astype(float))
    cols = [twoway_demean(dpost).ravel()]
    if early:
        cols.append(twoway_demean(np.outer(treat.astype(float), np.isin(yk, early).astype(float))).ravel())
    X = np.column_stack(cols); b = np.linalg.lstsq(X, twoway_demean(Yk).ravel(), rcond=None)[0]
    return b


def ri(Y, n_treat, observed_treat, years_used, ev_years=EVENT_YEARS, group_trend=False, max_perm=6000, **kw):
    """Exact randomisation inference: distribution of every event-study coefficient (and basic DID) under placebo assignments."""
    n = Y.shape[0]
    combos = list(itertools.combinations(range(n), n_treat))
    if len(combos) > max_perm: combos = [combos[i] for i in rng.choice(len(combos), max_perm, replace=False)]
    out_ev, out_bd = [], []
    for c in combos:
        t = np.zeros(n, bool); t[list(c)] = True
        out_ev.append(list(twfe(Y, t, years_used, ev_years, group_trend=group_trend).values()))
        out_bd.append(basic_did(Y, t, years_used))
    return np.array(out_ev), np.array(out_bd), len(combos)


def run(Y, names, treated_names, label, years_used, tag):
    n_treat = len(treated_names)
    treat = np.array([n in treated_names for n in names])
    obs_ev = np.array(list(twfe(Y, treat, years_used, EVENT_YEARS).values()))
    obs_bd = basic_did(Y, treat, years_used)
    perm_ev, perm_bd, B = ri(Y, n_treat, treat, years_used)
    p_ev = (np.abs(perm_ev) >= np.abs(obs_ev) - 1e-12).mean(0)
    p_bd = (np.abs(perm_bd) >= np.abs(obs_bd) - 1e-12).mean(0)
    # treated-specific linear trend (robustness)
    gt = twfe(Y, treat, years_used, [2020, 2021, 2022], group_trend=True)
    perm_gt = np.array([list(twfe(Y, np.isin(np.arange(len(names)), c), years_used, [2020, 2021, 2022], group_trend=True).values())
                        for c in itertools.islice(itertools.combinations(range(len(names)), n_treat), 0, None, max(1, int(np.ceil(
                            len(list(itertools.combinations(range(len(names)), n_treat))) / 2000))))])
    p_gt = (np.abs(perm_gt) >= np.abs(np.array(list(gt.values()))) - 1e-12).mean(0)
    ev = pd.DataFrame({"year": EVENT_YEARS, "beta": obs_ev, "pct": (np.exp(obs_ev) - 1) * 100,
                       "ri_p": p_ev, "placebo_lo": np.quantile(perm_ev, .025, 0), "placebo_hi": np.quantile(perm_ev, .975, 0)})
    ev.round(4).to_csv(tabpath(f"did_event_study_{tag}.csv"), index=False, encoding="utf-8-sig")
    summ = pd.DataFrame([
        {"spec": "Treat x post(2021-22)", "beta": obs_bd[0], "pct": (np.exp(obs_bd[0]) - 1) * 100, "ri_p": p_bd[0], "n_perm": B},
        {"spec": "Treat x 2020", "beta": obs_bd[1], "pct": (np.exp(obs_bd[1]) - 1) * 100, "ri_p": p_bd[1], "n_perm": B},
        {"spec": "w/ treated linear trend: Treat x 2021", "beta": gt[2021], "pct": (np.exp(gt[2021]) - 1) * 100, "ri_p": p_gt[1], "n_perm": len(perm_gt)},
        {"spec": "w/ treated linear trend: Treat x 2022", "beta": gt[2022], "pct": (np.exp(gt[2022]) - 1) * 100, "ri_p": p_gt[2], "n_perm": len(perm_gt)},
        {"spec": "w/ treated linear trend: trend (per yr)", "beta": gt["trend"], "pct": (np.exp(gt["trend"]) - 1) * 100, "ri_p": p_gt[3], "n_perm": len(perm_gt)},
    ]).round(4)
    summ.insert(0, "design", label)
    summ.to_csv(tabpath(f"did_summary_{tag}.csv"), index=False, encoding="utf-8-sig")
    print(f"\n=== {label} ({B} placebo assignments) ===")
    print(summ.to_string(index=False)); print(ev.round(3).to_string(index=False))
    # event-study figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(ev.year, ev.placebo_lo * 100, ev.placebo_hi * 100, color="grey", alpha=.25, label="安慰劑分布 2.5–97.5%")
    ax.plot(ev.year, ev.beta * 100, "o-", c="tab:red", label="處理組 − 對照組（log points ×100）")
    ax.axhline(0, c="k", lw=.7); ax.axvline(2020.5, ls="--", c="grey"); ax.text(2020.6, ax.get_ylim()[1] * .85, "疫情後", fontsize=9)
    ax.axvline(BASE + .01, ls=":", c="grey"); ax.grid(alpha=.3); ax.legend(loc="lower left")
    ax.set(title=f"事件研究：{label}（基期 {BASE}）", xlabel="年", ylabel="差異 (log points ×100)")
    fig.savefig(figpath(f"did_event_study_{tag}.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
    return ev, summ


# ====================== Part A: counties ======================
d = load_panel()
W = d[d.year.isin(YEARS)].pivot(index="county", columns="year", values="ln_asir")
names = W.index.to_list(); Y = W.values
run(Y, names, ["台北市", "新北市"], "縣市DID：雙北（台北＋新北）vs 其餘19縣市", YEARS, "counties_taipei")
run(Y, names, ["台北市", "新北市", "基隆市", "桃園市"], "縣市DID：北北基桃 vs 其餘17縣市", YEARS, "counties_north4")

# unadjusted means for the 2x2 table (treated = Taipei + New Taipei, ASIR levels)
tt = d[d.county.isin(["台北市", "新北市"])].groupby("year").apply(lambda g: np.average(g.asir, weights=g["pop"]))
cc = d[~d.county.isin(["台北市", "新北市"])].groupby("year").apply(lambda g: np.average(g.asir, weights=g["pop"]))
pd.DataFrame({"treated_Taipei_NewTaipei": tt, "control_other19": cc}).loc[2015:2022].round(2).to_csv(
    tabpath("did_group_means_2015_2022.csv"), encoding="utf-8-sig")
print("\npopulation-weighted ASIR, treated vs control:\n", pd.DataFrame({"treated": tt, "control": cc}).loc[2017:2022].round(1).T.to_string())

# ====================== Part B: cancer sites (national) ======================
nat = pd.read_csv(NATIONAL)
nat = nat[nat.year.isin(YEARS)]
SCREENED = [("結直腸", "全"), ("口腔、口咽及下咽", "全"), ("女性乳房", "女"), ("子宮頸", "女")]
CONTROL = [("胃", "全"), ("肝及肝內膽管", "全"), ("胰", "全"), ("食道", "全"), ("膀胱", "全"), ("非何杰金氏淋巴瘤", "全"),
           ("白血病", "全"), ("腦", "全"), ("攝護腺", "男")]
LUNG = [("肺、支氣管及氣管", "全")]
def series(lst):
    out = {}
    for s, sx in lst:
        x = nat[(nat.site == s) & (nat.sex == sx)].set_index("year")["asir"].reindex(YEARS)
        if x.notna().all() and (x > 0).all(): out[f"{s}({sx})"] = np.log(x.values)
    return out
S, C, L = series(SCREENED), series(CONTROL), series(LUNG)
print("\nscreened:", list(S), "\ncontrol:", list(C))
names_b = list(S) + list(C); Yb = np.array([*S.values(), *C.values()])
run(Yb, names_b, list(S), "癌別DID：篩檢導向癌別（大腸、口腔、乳、子宮頸）vs 非篩檢癌別", YEARS, "sites_screened")
names_l = list(L) + list(C); Yl = np.array([*L.values(), *C.values()])
run(Yl, names_l, list(L), "癌別DID：肺癌 vs 非篩檢癌別", YEARS, "sites_lung")
