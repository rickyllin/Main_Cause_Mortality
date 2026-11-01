# notebooks/ — Jupyter 筆記本

`../src/` 是正式、可重現的流程；這裡是「探索與學習用」的筆記本。原本 Colab 上 191 cells 的單一大檔，
已依章節拆成 9 個小的子任務筆記本，每個檔案第一格都標明**用途／輸入／輸出／來源 cell 範圍**。

## final_analysis/ — 最終版（Update 05/29）拆分，可在本機依序執行
每本第二格 `from _legacy_setup import *` 會提供 `df`、`pivot_table_age_std`、`pivot_table_cleaned`、`scaled_data`、`taiwan_map`、字型等變數
（取代原 Colab 的 Drive 掛載／字型下載／GDAL 設定）。圖與中間檔寫入 `notebooks/_scratch/`（git 不追蹤）。

| 筆記本 | 用途 | 對應 src 腳本 |
|---|---|---|
| `01_load_and_preprocess` | 載入 CSV、欄位改名、篩選肺癌、建縣市×年矩陣、相關係數 | 01 |
| `02_eda_correlation` | 縣市相關熱圖（階層式排序）、折線圖、箱型圖 | 02 |
| `03_cleaning_and_pca_corr` | 缺值檢查、排除連江縣、標準化、PCA（相關矩陣）、陡坡／累積／雙標圖 | 03 |
| `04_pca_cov_and_comparison` | PCA（共變異數矩陣）與相關矩陣版本比較 | 03 |
| `05_kmeans` | K-means：手肘法、輪廓係數、k=2..5 比較 | 04 |
| `06_hierarchical_clustering` | 4 種連結法 × k=2..10 輪廓係數、k=2..5 分群大表、樹狀圖 | 04 |
| `07_cluster_maps` | 分群地圖（需 geopandas） | 05 |
| `08_cluster_trend_lines` | 各群平均趨勢與增長摘要 | 05 |
| `09_appendix_old_code` | 舊版程式碼對照（不保證適用新流程） | — |

> 原稿 cell 106 把 pivot 轉成「年×縣市」，與下一格變數名稱相反，已移除；本資料夾的矩陣一律是「縣市×年」。
> 驗證：01–09 皆已在本機以乾淨環境完整執行，無錯誤。

## archive/ — 歷史版本（供追溯，不保證可獨立執行）
| 檔案 | 內容 | 狀態 |
|---|---|---|
| `A_early_version_0519` | 最早期草稿（中文欄位、未排除連江縣） | 執行至約 cell 26，之後缺舊變數 |
| `B_version_0523_pca` | 05/23 版 PCA（PC1–PC4 負荷量與得分） | 僅 1 處缺變數（loadings） |
| `C_version_0523_clustering` | 05/23 版分群（Pearson／Spearman 距離、Scipy vs Sklearn） | 部分 cell 缺舊變數 |
| `D_early_outline_draft` | 期末報告早期章節構想＋PCA／分群雛形 | 部分 cell 缺舊變數 |
| `E_course_submission_version` | 課程提交版（精簡 59 cells） | 部分 cell 依賴 Colab 當時的執行順序 |

## 執行方式
```bash
pip install -r ../requirements.txt jupyter
jupyter lab   # 從 notebooks/final_analysis/ 依編號執行；_legacy_setup.py 會自動找到專案根目錄
```
