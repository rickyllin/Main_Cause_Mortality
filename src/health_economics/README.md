# src/health_economics/ — 健康經濟學案例探討（程式）

與主流程（`../01–09`）分開的第二組分析：把同一份資料當作**面板**，用計量經濟學與因果推論方法回答三個問題：
長期趨勢（固定效果）、2021 年疫情衝擊（DID／合成控制／DML）、健康不平等（不平等指標＋空間自回歸）。
報告：[`../../paper/health_economics.tex`](../../paper/health_economics.tex)；輸出：[`../../outputs/health_economics/`](../../outputs/health_economics/)。

```bash
pip install -r ../../requirements.txt        # 需要 statsmodels、joblib、scikit-learn
cd src/health_economics
python 10_panel_prep.py && python 11_fixed_effects.py && python 12_did_covid.py \
  && python 13_synthetic_control.py && python 14_dml_did.py && python 15_parallel_trends_sensitivity.py \
  && python 16_spatial_inequality.py          # 16 需要先跑過 ../09_spatial_clustering.py（鄰接矩陣）
```
（`make health_econ` 等同上列指令。）

| 腳本 | 用途 | 方法 | 主要輸出 |
|---|---|---|---|
| `he_common.py` | 共用設定：路徑、輸出主題資料夾（`fe`／`did`／`causal`／`inequality`） | — | — |
| `10_panel_prep.py` | 建立縣市×年面板（ASIR、個案數、由粗率推算的人口、口腔癌／子宮頸癌風險代理）與全國癌別面板 | — | `data/processed/panel_*.csv` |
| `11_fixed_effects.py` | 長期趨勢：共同趨勢、二次趨勢、縣市別斜率與等斜率 F 檢定、雙向 FE 年度效應與變異分解、σ／β 收斂、時變共變量 | 固定效果（縣市叢集標準誤、WLS） | `fe/` |
| `12_did_covid.py` | 2021 疫情衝擊：縣市 DID（雙北、北北基桃）、癌別 DID（篩檢導向 vs 非篩檢；肺癌 vs 非篩檢）、事件研究、處理組專屬趨勢 | 雙向 FE DID＋**精確隨機化推論**（遍歷所有處理指派） | `did/` |
| `13_synthetic_control.py` | 雙北反事實：合成控制與嶺迴歸增強（ASCM）、空間安慰劑、不同捐贈池 | SC／ASCM | `causal/sc_*` |
| `14_dml_did.py` | 縣市×癌別 473 條序列的雙重穩健 DID：交叉配適（以縣市分組）的 GBM／Lasso／RF，OR-DML 與 DR-DML，隨機化推論、異質性、事前安慰劑 | **Double/debiased ML** | `causal/dml_*` |
| `15_parallel_trends_sensitivity.py` | 平行趨勢違反的敏感度（簡化版相對大小限制，崩潰值 M） | Rambachan–Roth 風格 | `causal/sens_*` |
| `16_spatial_inequality.py` | 不平等指標（Gini、Theil、CV、SII、RII、集中指數）；9 個五年窗口的 SAR／SEM（ML）、LM 檢定、Moran's I；空間 β 收斂 | 空間計量 | `inequality/` |

## 注意事項
- **處理組只有 2 個縣市**（台北市、新北市）：叢集穩健標準誤不可靠，所有 DID／DML 的 p 值皆為隨機化推論（210 種指派；北北基桃為 5985 種）。
- DML 的角色是「高維度共變量下的穩健調整」，**不能**彌補處理單位太少或未觀察混雜；請與 SC、DID 並讀。
- 資料沒有所得、教育、醫療資源與縣市別 COVID 病例數；人口規模只能當作都會梯度的代理。
- 2021 年雙北為疫情核心的設定依據新聞報導，請以疾管署開放資料核對（報告中有【待確認】標註）。
- `14_dml_did.py` 約需數分鐘（已限制 BLAS 執行緒並以 4 個工作者平行）。
