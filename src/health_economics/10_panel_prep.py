"""Build the analysis panels for the health-economics case study.

panel_lung_county_year.csv      county x year (lung cancer, both sexes, Lienchiang excluded):
                                asir, n (cases), crude, pop (= n / crude x 1e5), ln_asir, ln_pop,
                                oral_asir / cervix_asir  (risk-factor proxies from other cancer sites)
panel_cancer_national_year.csv  national-level ASIR and cases for every cancer site x sex x year (for the screening DID)
"""
import numpy as np
import pandas as pd

from he_common import EXCLUDED, LUNG, PANEL, NATIONAL, RAW_CSV, setup_plot

setup_plot()
raw = pd.read_csv(RAW_CSV)
raw.columns = ["year", "sex", "county", "site", "asir", "n", "mean_age", "median_age", "crude"]
raw["n"] = pd.to_numeric(raw["n"].astype(str).str.replace(",", "").str.strip())

both = raw[raw.sex == "全"]
lung = both[(both.site == LUNG) & (both.county != "全國") & (~both.county.isin(EXCLUDED))].copy()
lung["pop"] = lung.n / lung.crude * 1e5
lung["ln_asir"], lung["ln_pop"] = np.log(lung.asir), np.log(lung["pop"])
lung["aging_index"] = lung.crude / lung.asir            # crude / standardised: larger = older age structure

def proxy(site, sex, name):
    d = raw[(raw.site == site) & (raw.sex == sex) & (raw.county != "全國")]
    return d.set_index(["county", "year"])["asir"].rename(name)
# risk-factor proxies: oral cancer (tobacco / betel-nut / alcohol), cervical cancer (screening & socio-economic gradient, women)
for site, sex, name in [("口腔、口咽及下咽", "全", "oral_asir"), ("子宮頸", "女", "cervix_asir")]:
    lung = lung.merge(proxy(site, sex, name).reset_index(), on=["county", "year"], how="left")

cols = ["county", "year", "asir", "n", "crude", "pop", "ln_asir", "ln_pop", "aging_index", "mean_age",
        "oral_asir", "cervix_asir"]
lung = lung.sort_values(["county", "year"])[cols]
lung.to_csv(PANEL, index=False, encoding="utf-8-sig")
print(lung.shape, "| counties:", lung.county.nunique(), "| years:", lung.year.min(), "-", lung.year.max())
print("missing:", lung.isna().sum()[lambda s: s > 0].to_dict())

nat = raw[raw.county == "全國"][["year", "sex", "site", "asir", "n"]]
nat.to_csv(NATIONAL, index=False, encoding="utf-8-sig")
print("national panel:", nat.groupby(["site", "sex"]).ngroups, "site x sex series;", nat.year.min(), "-", nat.year.max())
