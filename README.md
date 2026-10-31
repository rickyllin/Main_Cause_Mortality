# Main_Cause_Mortality

台灣 21 縣市肺癌（肺、支氣管及氣管）**年齡標準化率**長期追蹤分析（1979–2022，政大碩二〈多變量分析〉期末報告）。
以 PCA 萃取全島共同趨勢與區域分化，再以 K-means 與階層式分群（single / complete / average / ward）合併相似縣市，並繪製地理分布圖。

## 主要結果
| 項目 | 結果 |
|---|---|
| PCA（相關係數矩陣） | PC1 53.6%、PC2 13.6%、前 4 個 PC 累積 82.3%、前 7 個 90.8% |
| PCA（共變異數矩陣） | PC1 54.1%、PC2 16.0%（與相關矩陣版本結論一致） |
| K-means | silhouette 最佳 k=2，次佳 k=5 |
| 階層式分群 | k=2 各 linkage 最佳（0.46）；Ward 與 K-means 在 k=2、5 的分群相近 |

詳細圖表見 `outputs/`，完整論述見 `reports/lung_cancer_longitudinal_analysis.docx`。

## 專案結構
```
data/raw/        原始資料（癌症登記，各癌別 × 縣市 × 性別 × 年）
data/external/   內政部縣市界線 shapefile (1140318)
data/processed/  縣市 × 年份矩陣（由 01 產生）
src/             可重現的分析流程（01→05）
outputs/         figures/ 圖、tables/ 表（由腳本產生）
reports/         期末報告 docx、題目發想紀錄
notebooks/legacy 原始 Colab notebooks（已清除輸出，僅供追溯）
```

## 執行
```bash
pip install -r requirements.txt
cd src
python 01_preprocess.py   # 篩選肺癌/全性別、pivot、排除連江縣
python 02_eda.py          # 趨勢、箱型圖、縣市相關熱圖
python 03_pca.py          # corr vs cov PCA、scree、biplot
python 04_cluster.py      # K-means / 階層式分群、輪廓係數、樹狀圖
python 05_maps.py         # 分群地圖 + 群組平均趨勢
```
繁體中文字型自動偵測（PingFang TC / Heiti TC / Noto Sans CJK 等）。

## 方法重點
- 資料：縣市（21，排除連江縣：11 個年份缺值、案例數極少）× 年（44）。
- 各年度跨縣市標準化後再做 PCA／分群（等同相關矩陣 PCA），聚焦趨勢型態而非絕對水準。
- 縣市相關性須用 `pivot.T.corr()`（`pivot.corr()` 算的是年份間相關）。

## 已知限制 / 修正紀錄
- **指標名稱已修正**：CSV 為癌症登記的年齡標準化**發生率**（非死亡率）。`reports/lung_cancer_longitudinal_analysis.docx` 內文已由「死亡率／死因統計」改為「發生率／癌症登記」（文獻回顧中引用的真實死因統計不動）。報告內嵌圖片裡的文字（如圖標題）無法自動修改，需回原圖重製。若要做真正的死因（死亡）追蹤，需改用衛福部死因統計資料。
- **縣市合併說明已修正**：資料縣市別本就是現行區劃，無需加權合併；報告已改為說明此點。
- 澎湖、金門人口少，年度波動大，會單獨成群，分群結果對其敏感。
- 原始作業 HW1 notebook 與課程中間稿未納入；`notebooks/legacy/` 內的 markdown 仍保留舊的「死亡率」措辭。
