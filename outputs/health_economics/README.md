# outputs/health_economics/ — 健康經濟學案例的輸出

由 `src/health_economics/10–16` 產生；報告 `paper/health_economics.tex` 直接引用。

| 主題 | 腳本 | 圖 | 表 |
|---|---|---|---|
| `fe`（固定效果） | 11 | `fe_year_effects`、`fe_county_slopes`、`fe_beta_convergence` | `fe_models`（所有係數）、`fe_county_specific_slopes`、`fe_year_effects`、`fe_variance_decomposition`、`fe_sigma_convergence`、`fe_beta_convergence` |
| `did`（DID） | 12 | `did_event_study_{counties_taipei,counties_north4,sites_screened,sites_lung}` | `did_summary_*`、`did_event_study_*`、`did_group_means_2015_2022` |
| `causal`（合成控制、DML、敏感度） | 13、14、15 | `sc_fit`、`sc_placebo_gaps`、`dml_ri_distribution`、`dml_did_by_site` | `sc_summary`、`sc_gap_paths`、`sc_donor_weights`、`dml_did_att`、`dml_did_naive`、`dml_did_heterogeneity`、`dml_did_by_site`、`dml_did_placebo_time`、`sens_parallel_trends` |
| `inequality`（不平等、空間迴歸） | 16 | `ineq_trends`、`sar_rho_by_window` | `ineq_indices_by_year`、`sar_models_by_window`、`sar_spatial_convergence` |

效果值單位為 log points（×100 約為百分比）；`ri_p` 為精確隨機化推論 p 值。
