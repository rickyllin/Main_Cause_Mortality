# outputs/ — 分析輸出（由 src/ 腳本產生，可重現）

依主題分資料夾；每個主題底下 `figures/<topic>/` 為圖、`tables/<topic>/` 為表。

| 主題 | 產生腳本 | 圖（重點） | 表（重點） |
|---|---|---|---|
| `eda` | 02 | `eda_trend_lines`（各縣市趨勢）、`eda_boxplot_counties`、`eda_correlation_heatmap` | `county_correlation.csv` |
| `pca` | 03 | `pca_scree_cumulative`（陡坡／累積）、`pca_biplot_corr`（縣市=點、年度=箭頭） | `pca_explained_variance`、`pca_corr_loadings_years`、`pca_corr_scores_counties` |
| `clustering` | 04、05、07 | `kmeans_*`（肘部／輪廓／PCA 散佈）、`hclust_silhouette`、`dendrogram_*`、`map_trend_*`（分群地圖＋群趨勢）、`more_cluster_agreement_stability`（ARI 與穩定性）、`more_cluster_membership_k5`、`fuzzy_membership_k3` | `kmeans_cluster_assignments`、`hclust_cluster_assignments`、`more_cluster_assignments`、`more_cluster_ari_vs_ward`、`more_cluster_bootstrap_stability`、`gmm_bic`、`fuzzy_membership_k*`、`growth_curve_features` |
| `bayes` | 08 | `bayes_forest_2018_22`（原始 vs 後驗＋可信區間）、`bayes_trajectories`、`bayes_coclustering_k5` | `bayes_county_summary`、`bayes_hyperparameters`、`bayes_coclustering_prob_k5` |
| `spatial` | 09 | `spatial_ward_map_k3/4/5`、`moran_i_by_year` | `moran_i`、`moran_i_by_year`、`spatial_cluster_assignments`、`spatial_vs_unconstrained`、`county_adjacency` |
| `mortality` | 06 | `incidence_vs_mortality_scatter`、`mortality_by_county_year` | `lung_mortality_county_year`、`lung_mir_county_year`、`lung_mir_county_mean_2020_2022` |

分群標籤統一為 `Group A, B, C…`（依縣市名稱出現順序，**不同方法的字母不可直接對應**）；比較請用 ARI（`more_cluster_ari_vs_ward.csv`）。
論文 `paper/main.tex` 直接引用這些圖。
