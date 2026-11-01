# src/ — 分析程式（可重現流程）

所有腳本依編號順序執行，彼此以 `data/processed/` 與 `outputs/tables/` 內的檔案銜接。
共用設定（路徑、字型、縣市排除清單、隨機種子、輸出路徑函式）在 [`common.py`](common.py)。

```bash
pip install -r ../requirements.txt
cd src
python 01_preprocess.py          # 依序執行；或在專案根目錄 make analysis
```

| 腳本 | 用途 | 主要輸入 | 主要輸出（`outputs/` 下） |
|---|---|---|---|
| `01_preprocess.py` | 篩選「肺癌／全性別／各縣市」，轉成縣市 × 年份矩陣；排除連江縣（11 個缺值年度） | `data/raw/lung_cancer_incidence_1979_2022.csv` | `data/processed/lung_age_std_county_year.csv` |
| `02_eda.py` | 趨勢折線圖、縣市箱型圖、縣市間相關係數熱圖 | processed 矩陣 | `figures/eda/`、`tables/eda/` |
| `03_pca.py` | PCA（相關矩陣 vs 共變異數矩陣）、陡坡圖、雙標圖、負荷量與得分 | processed 矩陣 | `figures/pca/`、`tables/pca/` |
| `04_cluster.py` | K-means（肘部／輪廓係數、PCA 散佈圖）與階層式分群（4 種連結法、樹狀圖） | processed 矩陣 | `figures/clustering/`、`tables/clustering/`（`kmeans_*`、`hclust_*`） |
| `05_maps.py` | 分群地圖＋各群平均趨勢（需 geopandas） | 04 的分群表、縣市界線 | `figures/clustering/map_trend_*.png` |
| `06_mortality.py` | 由衛福部各縣市死因統計表抽出肺癌（C33–C34）標準化死亡率；計算 MIR（死亡率÷發生率） | `data/raw/mohw_cause_of_death/*.xls` | `figures/mortality/`、`tables/mortality/` |
| `07_more_clustering.py` | 追蹤資料常用的其他分群：PAM、相關距離、DTW、譜分群、GMM、成長曲線混合、模糊 C 平均、DBSCAN；ARI 與 bootstrap 穩定性 | processed 矩陣、04 的分群表 | `figures/clustering/more_cluster_*`、`tables/clustering/more_cluster_*` 等 |
| `08_bayes_hierarchical.py` | 貝氏階層（部分合併）小區域估計：縣市水準／趨勢、95% 可信區間、排名機率、後驗同群機率 | processed 矩陣、原始 CSV 的個案數 | `figures/bayes/`、`tables/bayes/` |
| `09_spatial_clustering.py` | 縣市鄰接圖、Moran's I、空間約束 Ward 分群與地圖 | processed 矩陣、縣市界線 | `figures/spatial/`、`tables/spatial/` |

## 輸出命名與分類
`common.figpath(name)` / `common.tabpath(name)` 依檔名前綴把輸出放入主題資料夾
（`eda`、`pca`、`clustering`、`bayes`、`spatial`、`mortality`）。新增輸出時，只要沿用既有前綴（或在 `common._TOPIC_RULES` 新增規則）即可。

## 慣例
- 縣市名稱一律用「台」（不用「臺」）；界線檔中的「臺」在載入時轉換。
- 隨機種子固定為 `common.RANDOM_STATE = 42`，重跑結果一致。
- 標準化：每個年度（欄）在 21 縣市間標準化後再做 PCA／分群，所以分析的是「相對位置」，全國年度水準已被移除（詳見 `paper/main.tex` §2.3）。
