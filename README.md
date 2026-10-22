# HW2: 多元線性迴歸 (Multiple Linear Regression) 與特徵選擇分析

本專案為機器學習作業二，嚴格遵循 **CRISP-DM (Cross-Industry Standard Process for Data Mining)** 流程，使用公開 Kaggle 資料集建立多元線性迴歸模型，完成從商業理解、資料理解、資料準備、特徵選擇、模型評估、預測區間推論到 Web 部署之全流程。

---

## 🏆 作業要求與加分功能對照

| 項目 | 作業要求規範 | 本專案實作成效 |
| :--- | :--- | :--- |
| **加分功能 (+1 分)** | 有下拉視窗呈現兩種 Problem | ✅ Streamlit 側邊欄建立 `Problem ▼` 下拉選單，可即時切換 **1. Multiple Linear Regression (Baseline)** 與 **2. Feature Selection + Multiple Linear Regression**。 |
| **加分功能 (+2 分)** | 可任意輸入 Dataset CSV URL | ✅ Web UI 支援輸入任意公開 CSV 網址（如 `50_Startups.csv`），並具備健全防護與錯誤處理機制，不會因外部資料瑕疵造成系統 Crash。 |
| **資料來源** | Kaggle 公開資料集 (特徵數 > 13 且 < 40) | ✅ 採用 [Kaggle: House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)，包含 **21,613** 筆樣本與 **18** 個特徵（嚴格符合 14~39 特徵限制）。 |
| **分析任務** | 多元線性迴歸 (非分類任務) | ✅ 目標變數為連續價格 `price`（float64），完全不做 classification。 |
| **特徵選擇** | 必須執行 Feature Selection 並比較前後表現 | ✅ 實作 Lasso (L1 正則化) 與 RFE，明確列出選中特徵、被移除特徵與前後評估指標差異。 |
| **模型評估** | 計算 MSE, RMSE, MAE, R² 與殘差分析 | ✅ 完整產出模型比較表、殘差常態性分佈圖與等分散性診斷圖。 |
| **區間推論** | 預測圖加上信賴區間 (CI) 或 預測區間 (PI) | ✅ 依據統計學公式嚴謹推導槓桿值 $h_{00}$，同時繪製 95% Confidence Interval (CI) 與 95% Prediction Interval (PI)。 |
| **CRISP-DM** | 遵循六大步驟說明與實作 | ✅ Notebook 與 README 明確切分六大區塊。 |
| **AI 協助記錄** | 預留 GPT 輔助內容與 NotebookLM 摘要 | ✅ 於報告末尾獨立建立兩大專屬章節。 |

---

## 📂 專案目錄架構

```text
HW2 Multiple Linear regression/
│
├── data/
│   ├── .gitkeep
│   └── kc_house_data.csv          # Kaggle 原始房價資料集 (21,613 筆，18 特徵)
│
├── notebooks/
│   └── HW2_Multiple_Linear_Regression.ipynb   # 完整執行過之 CRISP-DM 分析筆記本
│
├── src/                           # 核心 Python 模組
│   ├── __init__.py
│   ├── data_loader.py             # 支援本地或任意 URL CSV 讀取與健全檢查 (Error Handling)
│   ├── preprocessing.py          # 缺失值處理、特徵剔除、標準化、Train/Test 切分
│   ├── feature_selection.py      # Lasso / RFE 特徵篩選模組
│   ├── model.py                  # 多元線性迴歸模型、嚴謹 CI/PI 區間計算推論
│   └── evaluation.py             # MSE, RMSE, MAE, R², 殘差分析圖表生成
│
├── app/
│   └── app.py                     # Streamlit Web 加分應用程式
│
├── figures/                       # 自動匯出之高解析圖表 (預測區間圖、殘差診斷圖)
│   ├── baseline_prediction_intervals.png
│   ├── fs_prediction_intervals.png
│   ├── baseline_residuals.png
│   └── fs_residuals.png
│
├── .gitignore                     # Git 忽略設定
├── README.md                      # 專案報告與說明文件
├── requirements.txt               # 相依套件清單
└── main.py                        # 終端機一鍵執行全流程腳本
```

---

## 🔄 CRISP-DM 六大分析階段

### 1. Business Understanding (商業理解)
* **核心問題**：不動產價值評估傳統仰賴鑑價師人工鑑價，耗時且主觀。
* **專案目標**：
  1. 建立客觀之**多元線性迴歸 (Multiple Linear Regression)** 定價模型，量化各特徵之邊際價格貢獻。
  2. 透過**特徵選擇 (Feature Selection)** 消除多重共線性變數，精簡評估維度。
  3. 產出 **95% 信賴區間 (CI)** 與 **95% 預測區間 (PI)**，為購屋者與放貸機構提供嚴謹的風險評估依據。

---

### 2. Data Understanding (資料理解)
* **資料來源**：[Kaggle: House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
* **樣本數量**：`21,613` 筆房屋交易紀錄。
* **目標變數**：`price`（平均 $540,088，中位數 $450,000，偏度 4.02，為典型右偏態連續數值）。
* **18 個原始特徵清單**：
  1. `bedrooms` (臥室數)
  2. `bathrooms` (衛浴數)
  3. `sqft_living` (室內坪數/面積, 與房價相關係數 $r=0.7020$)
  4. `sqft_lot` (土地面積)
  5. `floors` (樓層數)
  6. `waterfront` (是否面水景)
  7. `view` (景觀等級)
  8. `condition` (屋況評分)
  9. `grade` (建築設計等級, $r=0.6674$)
  10. `sqft_above` (地面上面積, $r=0.6056$)
  11. `sqft_basement` (地下室面積)
  12. `yr_built` (建成年份)
  13. `yr_renovated` (翻新年份)
  14. `zipcode` (郵遞區號)
  15. `lat` (緯度, $r=0.3070$)
  16. `long` (經度)
  17. `sqft_living15` (鄰近 15 戶平均室內面積, $r=0.5854$)
  18. `sqft_lot15` (鄰近 15 戶平均土地面積)
* **缺失值與重複值**：全欄位缺失值均為 0，無重複列。

---

### 3. Data Preparation (資料準備)
* **欄位篩選**：自動剔除無預測性之流水號 `id` 與交易時間 `date`。
* **缺失值填補**：數值特徵以中位數填補，類別特徵以眾數填補。
* **類別編碼**：One-Hot Encoding（`drop_first=True` 避免虛擬變數陷阱）。
* **資料集切分**：80% 訓練集 (17,290 筆)，20% 測試集 (4,323 筆)，固定 `random_state=42`。
* **特徵標準化 (StandardScaler)**：僅在訓練集進行 `fit` 並轉換測試集，**嚴防資料外洩 (Data Leakage)**。

---

### 4. Modeling (模型建立)
建立兩組多元線性迴歸模型進行對照實驗：

#### Model 1: Baseline Multiple Linear Regression
* 使用全部 18 個原始特徵進行 OLS 最小平方法擬合。

#### Model 2: Feature Selection + Multiple Linear Regression
* **特徵選擇方法**：5-Fold 交叉驗證 **Lasso (L1 正則化)**。
* **特徵篩選結果**：
  * **原始特徵數**：18 個
  * **選中保留特徵 (17 個)**：`bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `grade`, `sqft_above`, `yr_built`, `yr_renovated`, `zipcode`, `lat`, `long`, `sqft_living15`, `sqft_lot15`
  * **被移除特徵 (1 個)**：`sqft_basement`（因 `sqft_living = sqft_above + sqft_basement` 具有嚴重的多重共線性，Lasso 自動將其係數收縮至 0）。

---

### 5. Evaluation (模型評估與殘差分析)

#### 5.1 模型指標對比表

| 評估指標 (Metric) | Baseline Model (18 特徵) | Feature Selected Model (17 特徵) | 差異 (FS - Base) |
| :--- | :---: | :---: | :---: |
| **R² (判定係數)** | **0.7012** | **0.7012** | **-0.0000** |
| **Adjusted R² (調整後 R²)** | **0.7000** | **0.7000** | **+0.0000** |
| **RMSE (均方根誤差)** | **$212,521.77** | **$212,539.52** | **+$17.75** (微乎其微) |
| **MAE (平均絕對誤差)** | **$127,369.11** | **$127,493.34** | **+$124.23** |
| **MSE (均方誤差)** | **4.52 × 10¹⁰** | **4.52 × 10¹⁰** | **+7.54 × 10⁶** |

> **結論**：剔除冗餘特徵後，模型複雜度降低，但 **$R^2$ 完全維持在 0.7012**，成功達到「模型精簡化」且「預測準確度不減」的雙重目標！

#### 5.2 預測圖與信賴/預測區間 (CI & PI)
嚴謹依據統計公式計算：
* **95% 信賴區間 (Confidence Interval)**：$\\hat{y}_0 \\pm t_{\\alpha/2, n-p} \\cdot s \\sqrt{x_0 (X^T X)^{-1} x_0^T}$
* **95% 預測區間 (Prediction Interval)**：$\\hat{y}_0 \\pm t_{\\alpha/2, n-p} \\cdot s \\sqrt{1 + x_0 (X^T X)^{-1} x_0^T}$

預測圖已匯出於 `figures/fs_prediction_intervals.png`。

#### 5.3 殘差診斷 (Residual Diagnostics)
* **常態性檢查**：殘差分佈大致呈現以 0 為中心的對稱鐘形分佈（匯出於 `figures/fs_residuals.png`）。
* **等分散性檢查**：絕大多數樣本殘差在配適值各區間均勻散佈，符合線性迴歸基本假設。

---

### 6. Deployment (Web 應用程式部署)

本專案使用 **Streamlit** 建立高互動性 Web 應用程式（位於 `app/app.py`）：

#### 💡 加分功能說明：
1. **下拉選單呈現兩種 Problem (+1 分)**：
   * 側邊欄頂部提供 `Problem ▼` 選單，可無縫切換檢視 **1. Baseline Multiple Linear Regression** 或 **2. Feature Selection + Multiple Linear Regression**。
2. **可任意輸入 Dataset CSV URL (+2 分)**：
   * 側邊欄提供網址輸入框，支援讀取任意網路公開 CSV（例如預設提供的 `https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv`）。
   * 內建防護罩機制：自動檢查 URL 連線狀態、CSV 格式、樣本數是否足夠、是否具備數值型目標變數；若不符合要求會給出提示，不會 Crash。
3. **即時互動房價預測器**：
   * 使用者可在網頁上透過滑桿調整特徵，模型即時推算出預測房價，並輸出 95% 信賴區間 (CI) 與 預測區間 (PI)。

---

## 🚀 執行指南 (How to Run)

### 1. 安裝環境相依套件
```bash
pip install -r requirements.txt
```

### 2. 終端機一鍵執行全流程
```bash
python main.py
```
> 將自動完成資料載入、前處理、特徵篩選、兩模型訓練評估，並將圖表輸出至 `figures/`。

### 3. 啟動 Streamlit Web 應用程式
```bash
streamlit run app/app.py
```
> 開啟瀏覽器存取 `http://localhost:8501` 即可體驗完整互動系統！

---

## 🤖 AI 協助要求紀錄

### GPT 輔助內容
* **提問與輔助歷程**：
  * 本次作業在 ChatGPT / Antigravity 輔助下，梳理 CRISP-DM 流程架構，並針對 Kaggle 公開資料集規格進行特徵數（14~39 特徵）之過濾與確認。
  * 協助推導多元線性迴歸槓桿值 ($h_{00} = x_0 (X^T X)^{-1} x_0^T$) 與信賴/預測區間 (CI/PI) 之矩陣運算公式。
  * 協助編寫 Streamlit Web App 之例外處理機制，防止非預期格式之 CSV 造成程式異常崩潰。
* **對話紀錄 PDF 附件**：
  * 完整的對話紀錄已完整匯出為 PDF 檔案，儲存於專案庫中：[`docs/GPT_Assistance_Log.pdf`](docs/GPT_Assistance_Log.pdf)。
  * 亦可於網頁直接檢視其原始排版文件：[`docs/GPT_Assistance_Log.html`](docs/GPT_Assistance_Log.html)。

---

### NotebookLM 摘要
> **【基於 NotebookLM 對多元線性迴歸與房價預測文獻之研究摘要】**
>
> 透過 NotebookLM 彙整 Kaggle 頂尖方案與不動產計量經濟學文獻（Hedonic Pricing Model），本研究針對房價多元線性迴歸得出以下三大核心洞察：
> 
> 1. **特徵結構與地段主導性**：在 King County 房價數據中，房屋總面積（`sqft_living`）、施工建造評級（`grade`）與地理緯度（`lat`）為解釋房價變異的核心驅動因子。其中緯度特徵精確捕捉了西雅圖都會區的就業重心與高資產聚集效應，證實了不動產「地段為王」的計量特質。
> 2. **多重共線性之處置價值**：文獻與社群廣泛指出，資料集中存在明確的結構重疊（例如室內總面積等於地上層加上地下室面積）。若僅採傳統 OLS，易造成係數變異數膨脹（VIF 偏高）；而透過 Lasso (L1 正則化) 進行特徵選擇，能將冗餘特徵（如 `sqft_basement`）權重收縮至零，在保持判定係數 $R^2 \approx 0.70$ 不衰退的前提下大幅提升模型簡約度（Parsimony）與外推泛化穩定性。
> 3. **統計推論與區間估計之實務意義**：大部分機器學習方案僅著重於單點預測（Point Estimate），常低估極端豪宅市場的高變異風險。本分析透過槓桿值矩陣嚴謹推導 95% 預測區間（Prediction Interval），將隨機誤差與估計不確定性納入考量，為不動產估價與銀行房貸風險控管提供了具備堅實統計理論依據的信心邊界。
