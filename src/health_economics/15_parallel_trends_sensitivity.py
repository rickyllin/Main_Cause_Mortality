"""Sensitivity of the DID / event-study estimates to violations of parallel trends (simplified 'relative magnitudes' bound).

Following the idea of Rambachan & Roth (2023): assume the post-period violation of parallel trends is no larger than M times the
largest pre-period deviation observed in the event study,   |delta_post| <= M * max_pre |beta_pre| .
The identified set for each post-period effect is  beta_post +/- M * max_pre|beta_pre|.  The 'breakdown M' is the smallest M at which the
set contains zero (large = conclusion robust). This is a simplified, absolute-deviation version, not the full FLCI procedure."""
import numpy as np
import pandas as pd

from he_common import setup_plot, tabpath

setup_plot()
rows = []
for tag, label in [("counties_taipei", "縣市：雙北"), ("counties_north4", "縣市：北北基桃"), ("sites_screened", "癌別：篩檢導向"), ("sites_lung", "癌別：肺癌")]:
    ev = pd.read_csv(tabpath(f"did_event_study_{tag}.csv")).set_index("year")
    pre = ev.loc[2012:2018, "beta"]; mx = pre.abs().max()
    for y in (2020, 2021, 2022):
        b = ev.loc[y, "beta"]
        row = {"design": label, "post_year": y, "beta_logpts": b, "max_abs_pre": mx, "breakdown_M": abs(b) / mx if mx > 0 else np.inf}
        for M in (0, 0.5, 1, 2):
            row[f"set_lo_M{M}"] = b - M * mx; row[f"set_hi_M{M}"] = b + M * mx
        rows.append(row)
out = pd.DataFrame(rows).round(4); out.to_csv(tabpath("sens_parallel_trends.csv"), index=False, encoding="utf-8-sig")
print(out[["design", "post_year", "beta_logpts", "max_abs_pre", "breakdown_M"]].to_string(index=False))
