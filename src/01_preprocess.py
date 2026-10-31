"""Raw cancer-registry CSV -> county x year matrix (lung cancer, both sexes)."""
import pandas as pd

from common import EXCLUDED, LUNG, PROCESSED, RAW_CSV, RATE_COL, setup_plot

setup_plot()
df = pd.read_csv(RAW_CSV)
sub = df[(df["性別"] == "全") & (df["縣市別"] != "全國") & (df["癌症別"] == LUNG)]
pivot = sub.pivot_table(index="縣市別", columns="癌症診斷年", values=RATE_COL, aggfunc="mean")

nan_cells = pivot.isna().stack()
print("Missing cells:\n", nan_cells[nan_cells].index.tolist())
pivot.to_csv(PROCESSED / "lung_age_std_county_year_with_lienchiang.csv", encoding="utf-8-sig")

clean = pivot.drop(index=EXCLUDED)
assert not clean.isna().any().any()
clean.to_csv(PROCESSED / "lung_age_std_county_year.csv", encoding="utf-8-sig")
print(f"Analysis matrix: {clean.shape[0]} counties x {clean.shape[1]} years")
