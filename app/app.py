"""
HW2 Multiple Linear Regression & Feature Selection Web Application
Streamlit App with CRISP-DM workflow, bonus dropdown problem selection (+1 pt),
and arbitrary Dataset CSV URL loader with robust error handling (+2 pts).
"""

import sys
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure local imports work cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_dataset
from src.preprocessing import preprocess_data
from src.feature_selection import perform_feature_selection
from src.model import LinearRegressionWithIntervals
from src.evaluation import compute_metrics, compare_models, plot_actual_vs_predicted, plot_residuals


# Page configuration
st.set_page_config(
    page_title="多元線性迴歸與特徵選擇系統 (CRISP-DM)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1e3a8a; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.1rem; color: #4b5563; margin-bottom: 1.5rem; }
    .card { background-color: #f8fafc; border-radius: 8px; padding: 1.2rem; border-left: 5px solid #3b82f6; margin-bottom: 1rem; }
    .metric-box { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; text-align: center; }
</style>
""", unsafe_allow_html=True)


# Sidebar Configuration
st.sidebar.title("⚙️ 系統設定與加分功能")

# ----------------------------------------------------
# 加分功能 1 (+1 分): 下拉選單呈現兩種 Problem
# ----------------------------------------------------
st.sidebar.markdown("### 🎯 Problem 下拉選擇 (+1 分)")
problem_choice = st.sidebar.selectbox(
    "Problem ▼",
    options=[
        "1. Multiple Linear Regression (Baseline Model)",
        "2. Feature Selection + Multiple Linear Regression"
    ],
    index=0,
    help="選擇欲檢視與分析的迴歸建模問題情境"
)

# ----------------------------------------------------
# 加分功能 2 (+2 分): 可任意輸入 Dataset CSV URL
# ----------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌐 自訂資料集來源 (+2 分)")
data_source_mode = st.sidebar.radio(
    "資料集來源模式:",
    ["預設 Kaggle 資料集 (King County Houses)", "自訂 CSV 網址 (URL)", "上傳本地 CSV 檔案"]
)

DEFAULT_KC_PATH = "data/kc_house_data.csv"
SAMPLE_STARTUPS_URL = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv"

raw_df = None
data_load_msg = ""
data_is_valid = False
dataset_name = ""

if data_source_mode == "預設 Kaggle 資料集 (King County Houses)":
    dataset_name = "Kaggle King County House Sales (18 Features)"
    raw_df, data_is_valid, data_load_msg, stats = load_dataset(DEFAULT_KC_PATH, target_col="price")

elif data_source_mode == "自訂 CSV 網址 (URL)":
    url_input = st.sidebar.text_input(
        "輸入任意 Dataset CSV URL:",
        value=SAMPLE_STARTUPS_URL,
        help="支援任意公開 CSV 直連網址，例如 GitHub Raw CSV"
    )
    col_btn1, col_btn2 = st.sidebar.columns(2)
    with col_btn1:
        if st.button("帶入 50_Startups 範例"):
            url_input = SAMPLE_STARTUPS_URL
    with col_btn2:
        if st.button("帶入 King County 網址"):
            url_input = "https://raw.githubusercontent.com/Shreyas3108/house-price-prediction/master/kc_house_data.csv"

    if url_input:
        dataset_name = f"自訂 URL 資料集 ({url_input.split('/')[-1]})"
        with st.spinner("正在自遠端讀取並驗證資料集..."):
            raw_df, data_is_valid, data_load_msg, stats = load_dataset(url_input)
    else:
        st.sidebar.warning("請輸入有效的 CSV 網址。")

elif data_source_mode == "上傳本地 CSV 檔案":
    uploaded_file = st.sidebar.file_uploader("上傳 CSV 檔案", type=["csv"])
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            dataset_name = f"上傳檔案: {uploaded_file.name}"
            from src.data_loader import validate_dataset
            data_is_valid, data_load_msg, stats = validate_dataset(raw_df)
        except Exception as e:
            data_is_valid = False
            data_load_msg = f"檔案解析錯誤: {e}"

# UI 錯誤處理防護罩 (Error Handling without Crash)
if not data_is_valid or raw_df is None:
    st.error(f"❌ 資料載入失敗或不符合規範: {data_load_msg}")
    st.info("💡 提示: 請確保輸入的 URL 為直接指向 CSV 的網址，包含至少 30 筆樣本與 2 個數值型特徵。")
    st.stop()

# Target Selection
all_num_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
if not all_num_cols:
    st.error("❌ 錯誤: 資料集中未發現任何數值型欄位，無法進行線性迴歸。")
    st.stop()

default_target_idx = 0
if "price" in all_num_cols:
    default_target_idx = all_num_cols.index("price")
elif "Profit" in all_num_cols:
    default_target_idx = all_num_cols.index("Profit")

target_col = st.sidebar.selectbox("🎯 選擇目標變數 (Target Column):", options=all_num_cols, index=default_target_idx)

# Feature Selection Method
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 特徵選擇參數")
fs_method = st.sidebar.selectbox("特徵篩選演算法:", ["Lasso (L1 Regularization)", "RFE (遞迴特徵消除)"])
fs_method_key = "lasso" if "Lasso" in fs_method else "rfe"

test_ratio = st.sidebar.slider("測試集比例 (Test Size):", min_value=0.1, max_value=0.4, value=0.2, step=0.05)


# ----------------------------------------------------
# Main Content Header
# ----------------------------------------------------
st.markdown('<div class="main-title">📈 多元線性迴歸與特徵選擇系統</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">遵循 <b>CRISP-DM</b> 標準架構 ｜ 目前模式: <b>{problem_choice}</b> ｜ 資料來源: <b>{dataset_name}</b></div>', unsafe_allow_html=True)

# Process Pipeline
with st.spinner("正在進行特徵處理、模型訓練與信賴區間計算..."):
    X_train, X_test, y_train, y_test, feature_names, scaler, prep_info = preprocess_data(
        raw_df, target_col=target_col, test_size=test_ratio, random_state=42
    )
    
    # Model 1: Baseline
    base_model = LinearRegressionWithIntervals(fit_intercept=True)
    base_model.fit(X_train, y_train)
    y_pred_base, ci_low_base, ci_up_base, pi_low_base, pi_up_base = base_model.predict_intervals(X_test)
    base_metrics = compute_metrics(y_test, y_pred_base, p_features=len(feature_names))

    # Feature Selection
    sel_idx, sel_feats, rem_feats, imp_df, fs_details = perform_feature_selection(
        X_train, y_train, feature_names, method=fs_method_key, random_state=42
    )

    # Model 2: Feature Selected
    X_train_sel = X_train[:, sel_idx]
    X_test_sel = X_test[:, sel_idx]
    fs_model = LinearRegressionWithIntervals(fit_intercept=True)
    fs_model.fit(X_train_sel, y_train)
    y_pred_fs, ci_low_fs, ci_up_fs, pi_low_fs, pi_up_fs = fs_model.predict_intervals(X_test_sel)
    fs_metrics = compute_metrics(y_test, y_pred_fs, p_features=len(sel_feats))


# CRISP-DM Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Business Understanding",
    "2. Data Understanding",
    "3. Data Preparation",
    "4. Modeling",
    "5. Evaluation",
    "6. Deployment"
])


# ====================================================
# Tab 1: Business Understanding
# ====================================================
with tab1:
    st.subheader("1. 商業理解 (Business Understanding)")
    st.markdown("""
    #### 房價與市場估價分析之商業價值
    * **核心商業問題**：在不動產市場中，房屋的交易價格受到地段區位、建物坪數、格局配備、施工等級等多重維度影響。傳統估價仰賴估價師人工評估，耗時且主觀。
    * **專案目標**：透過**多元線性迴歸 (Multiple Linear Regression)** 建立客觀、可量化且具備統計推論支撐的自動化估價模型，並透過**特徵選擇 (Feature Selection)** 排除多重共線性變數，簡化鑑價流程。
    * **產出效益**：
      1. 提供消費者與房產仲介公允的價格預測基準。
      2. 透過**預測區間 (Prediction Interval)** 與 **信賴區間 (Confidence Interval)**，量化未確定性風險，防止資產高估所帶來的放貸或投資虧損。
    """)
    st.info("💡 遵照作業加分規範，本系統同時支援載入任意外部商業資料集（如新創公司利潤 50_Startups 等），提供通用迴歸決策支援。")


# ====================================================
# Tab 2: Data Understanding
# ====================================================
with tab2:
    st.subheader("2. 資料理解 (Data Understanding)")
    
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    col_d1.metric("總樣本數 (Rows)", f"{len(raw_df):,}")
    col_d2.metric("原始欄位數 (Columns)", f"{raw_df.shape[1]}")
    col_d3.metric("缺失值數量 (Missing)", f"{raw_df.isnull().sum().sum()}")
    col_d4.metric("目標變數 (Target)", f"{target_col}")

    st.markdown("#### 資料集預覽")
    st.dataframe(raw_df.head(8), use_container_width=True)

    st.markdown("#### 統計敘述摘要 (Descriptive Statistics)")
    st.dataframe(raw_df.describe().T, use_container_width=True)

    # EDA Visualizations
    st.markdown("#### 探索性資料分析圖表 (EDA Visualizations)")
    eda_col1, eda_col2 = st.columns(2)
    
    with eda_col1:
        st.markdown(f"**目標變數 `{target_col}` 分佈與偏度分析**")
        fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
        sns.histplot(raw_df[target_col], kde=True, color="#1e88e5", ax=ax_dist, bins=30)
        ax_dist.set_title(f"Target Distribution: {target_col} (Skewness: {raw_df[target_col].skew():.2f})", fontsize=11)
        st.pyplot(fig_dist)

    with eda_col2:
        st.markdown("**特徵與目標變數相關係數 (Correlation Top 10)**")
        numeric_sub = raw_df.select_dtypes(include=[np.number])
        corr_series = numeric_sub.corr()[target_col].drop(target_col).sort_values(ascending=False).head(10)
        fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
        corr_series.plot(kind="barh", color="#43a047", ax=ax_corr)
        ax_corr.set_title(f"Top Correlated Features with {target_col}", fontsize=11)
        st.pyplot(fig_corr)


# ====================================================
# Tab 3: Data Preparation
# ====================================================
with tab3:
    st.subheader("3. 資料前處理 (Data Preparation)")
    
    st.markdown("""
    為確保多元線性迴歸的準確性並嚴防**資料外洩 (Data Leakage)**，本系統實施以下標準化資料前處理步驟：
    """)
    st.markdown(f"""
    * **排除非預測性欄位**：自動識別並剔除交易 ID、時間戳記等無效特徵（已剔除: `{prep_info['dropped_columns']}`）。
    * **缺失值填補**：數值特徵以中位數 (Median) 填補，類別特徵以眾數 (Mode) 填補（共填補 `{prep_info['total_missing_imputed']}` 個缺失值）。
    * **類別特徵編碼**：採用 One-Hot Encoding（`drop_first=True` 避免虛擬變數陷阱）。
    * **資料集切分 (Train/Test Split)**：依比例 `{1-test_ratio:.0%}` 訓練集與 `{test_ratio:.0%}` 測試集切分（隨機種子固定以確保可復現性）。
    * **特徵標準化 (Standardization)**：使用 `StandardScaler` 將特徵轉換為平均值 0、標準差 1，僅在訓練集擬合 (fit) 並轉換 (transform) 測試集。
    """)
    
    st.success(f"前處理完成：訓練集樣本 {prep_info['train_samples']:,} 筆 ｜ 測試集樣本 {prep_info['test_samples']:,} 筆 ｜ 最終有效特徵 {prep_info['final_feature_count']} 個。")


# ====================================================
# Tab 4: Modeling
# ====================================================
with tab4:
    st.subheader("4. 模型建立與特徵選擇 (Modeling & Feature Selection)")

    st.markdown("### 🎯 目前檢視情境: " + problem_choice)

    if "Baseline" in problem_choice:
        st.info("ℹ️ 目前呈現 **Model 1: Baseline Multiple Linear Regression**，使用全部原始特徵進行迴歸擬合。")
        active_features = feature_names
        active_model = base_model
        active_y_pred = y_pred_base
        active_ci_low, active_ci_up = ci_low_base, ci_up_base
        active_pi_low, active_pi_up = pi_low_base, pi_up_base
    else:
        st.info(f"ℹ️ 目前呈現 **Model 2: Feature Selection + Multiple Linear Regression**，採用 **{fs_details['method']}** 進行篩選後重構模型。")
        active_features = sel_feats
        active_model = fs_model
        active_y_pred = y_pred_fs
        active_ci_low, active_ci_up = ci_low_fs, ci_up_fs
        active_pi_low, active_pi_up = pi_low_fs, pi_up_fs

    # Feature Selection Breakdown (Teacher explicitly required)
    st.markdown("#### 📋 特徵選擇細節 (Feature Selection Breakdown)")
    fcol1, fcol2, fcol3 = st.columns(3)
    fcol1.metric("原始特徵數量", f"{fs_details['original_count']}")
    fcol2.metric("選中特徵數量 (Selected)", f"{fs_details['selected_count']}")
    fcol3.metric("被移除特徵數量 (Removed)", f"{fs_details['removed_count']}")

    st.markdown(f"**選中保留的特徵**：`{', '.join(sel_feats)}`")
    if rem_feats:
        st.markdown(f"**被移除的冗餘特徵**：`{', '.join(rem_feats)}`")
    else:
        st.markdown("**被移除的冗餘特徵**：無（所有特徵均通過顯著性閾值）")

    st.markdown("#### 特徵係數/權重排行")
    st.dataframe(imp_df, use_container_width=True)


# ====================================================
# Tab 5: Evaluation
# ====================================================
with tab5:
    st.subheader("5. 模型評估與殘差分析 (Model Evaluation & Residual Analysis)")

    # Metrics Summary Cards
    st.markdown("#### 📊 模型指標摘要 (Model Evaluation Metrics)")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    if "Baseline" in problem_choice:
        disp_metrics = base_metrics
    else:
        disp_metrics = fs_metrics

    m_col1.metric("R² 判定係數", f"{disp_metrics['R2']:.4f}")
    m_col2.metric("Adjusted R² 調整後判定係數", f"{disp_metrics['Adjusted_R2']:.4f}")
    m_col3.metric("RMSE 均方根誤差", f"${disp_metrics['RMSE']:,.2f}")
    m_col4.metric("MAE 平均絕對誤差", f"${disp_metrics['MAE']:,.2f}")

    # Comparison Table
    st.markdown("#### ⚖️ Baseline vs. Feature Selected Model 比較表")
    comp_table = compare_models(base_metrics, fs_metrics)
    st.table(comp_table)

    # Prediction Plot with Intervals (Teacher required: Rigorous CI & PI)
    st.markdown("#### 📉 預測圖：實際值 vs 預測值 (含 95% 信賴區間 CI 與 預測區間 PI)")
    st.caption("依據多元線性迴歸統計推論公式嚴謹計算：信賴區間 (CI) 反映平均預測反應之不確定性；預測區間 (PI) 額外計入單一觀測值之隨機誤差變異。")
    
    fig_pred = plot_actual_vs_predicted(
        y_test, active_y_pred, active_ci_low, active_ci_up, active_pi_low, active_pi_up,
        title=f"{problem_choice} - Actual vs. Predicted with 95% CI & PI",
        sample_size=80
    )
    st.pyplot(fig_pred)

    # Residual Diagnostics
    st.markdown("#### 🔬 殘差分析 (Residual Diagnostics)")
    fig_resid = plot_residuals(y_test, active_y_pred, title_prefix=problem_choice.split(" ")[0])
    st.pyplot(fig_resid)


# ====================================================
# Tab 6: Deployment
# ====================================================
with tab6:
    st.subheader("6. 模型部署與即時預測 (Deployment & Interactive Predictor)")
    
    st.markdown("""
    使用者可透過以下互動介面調整數值，即時調用已訓練之多元線性迴歸模型獲得預測結果，並由系統給予嚴謹的 **95% 預測區間 (PI)** 與 **信賴區間 (CI)**。
    """)

    # Interactive form for inputs
    with st.form("prediction_form"):
        st.markdown("##### 🎛️ 輸入預測變數")
        input_values = {}
        
        # Determine features used in the active model
        feats_to_input = active_features
        input_cols = st.columns(3)
        
        # Slice original dataframe to get reasonable min/max/mean for sliders
        clean_df_num = raw_df.select_dtypes(include=[np.number])
        for idx, feat in enumerate(feats_to_input):
            col_target = input_cols[idx % 3]
            with col_target:
                if feat in clean_df_num.columns:
                    col_min = float(clean_df_num[feat].min())
                    col_max = float(clean_df_num[feat].max())
                    col_mean = float(clean_df_num[feat].mean())
                    step_val = (col_max - col_min) / 100.0 if col_max > col_min else 1.0
                    input_values[feat] = st.slider(
                        f"{feat}",
                        min_value=col_min,
                        max_value=col_max,
                        value=col_mean,
                        step=max(0.01, step_val)
                    )
                else:
                    input_values[feat] = st.number_input(f"{feat}", value=0.0)

        submit_btn = st.form_submit_button("🚀 進行價格預測 (Predict)")

    if submit_btn:
        input_arr = np.array([[input_values[f] for f in active_features]])
        
        # If model used scaler on all features, align scaler transform
        if scaler is not None:
            # Reconstruct full vector if active model is feature selected
            if len(active_features) < len(feature_names):
                full_input = np.zeros((1, len(feature_names)))
                for f_idx, f_name in enumerate(feature_names):
                    if f_name in input_values:
                        full_input[0, f_idx] = input_values[f_name]
                    elif f_name in clean_df_num.columns:
                        full_input[0, f_idx] = clean_df_num[f_name].mean()
                scaled_full = scaler.transform(full_input)
                scaled_input = scaled_full[:, sel_idx]
            else:
                scaled_input = scaler.transform(input_arr)
        else:
            scaled_input = input_arr

        pred_val, ci_l, ci_u, pi_l, pi_u = active_model.predict_intervals(scaled_input)

        st.success("### 預測結果")
        r_col1, r_col2, r_col3 = st.columns(3)
        r_col1.metric("預估目標價格 (Point Prediction)", f"${pred_val[0]:,.2f}")
        r_col2.metric("95% 信賴區間 (CI)", f"${ci_l[0]:,.2f} ~ ${ci_u[0]:,.2f}")
        r_col3.metric("95% 預測區間 (PI)", f"${pi_l[0]:,.2f} ~ ${pi_u[0]:,.2f}")
