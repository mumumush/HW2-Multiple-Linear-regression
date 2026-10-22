"""
Main Pipeline Runner
Executes CRISP-DM workflow from Data Loading to Model Comparison and Visualization.
"""

import os
import sys
import numpy as np
import pandas as pd

# Set UTF-8 encoding for Windows console compatibility
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.data_loader import load_dataset
from src.preprocessing import preprocess_data
from src.feature_selection import perform_feature_selection
from src.model import LinearRegressionWithIntervals
from src.evaluation import compute_metrics, compare_models, plot_actual_vs_predicted, plot_residuals


def run_pipeline(data_path: str = "data/kc_house_data.csv", target_col: str = "price"):
    print("=" * 70)
    print("  HW2: Multiple Linear Regression & Feature Selection Pipeline")
    print("=" * 70)

    # 1. Data Loading
    print("\n[Step 1 & 2] 載入資料集...")
    df, is_valid, msg, stats = load_dataset(data_path, target_col=target_col)
    if not is_valid:
        print(f"錯誤: {msg}")
        sys.exit(1)

    print(f"-> 成功載入! 樣本數: {stats['rows']:,}, 總欄位數: {stats['cols']}")
    print(f"-> 缺失值總數: {stats['missing_cells']} ({stats['missing_ratio']:.2f}%)")

    # 2. Preprocessing
    print("\n[Step 3] 執行資料前處理 (Data Preparation)...")
    X_train, X_test, y_train, y_test, feature_names, scaler, prep_info = preprocess_data(
        df, target_col=target_col, test_size=0.2, random_state=42
    )
    print(f"-> 訓練樣本數: {len(X_train):,}, 測試樣本數: {len(X_test):,}")
    print(f"-> 原始特徵數量: {len(feature_names)}")
    print(f"-> 原始特徵列表: {', '.join(feature_names)}")

    # 3. Model 1: Baseline Multiple Linear Regression
    print("\n[Step 4] 建立 Model 1: Baseline Multiple Linear Regression (全特徵)...")
    base_model = LinearRegressionWithIntervals(fit_intercept=True)
    base_model.fit(X_train, y_train)

    y_pred_base, ci_low_base, ci_up_base, pi_low_base, pi_up_base = base_model.predict_intervals(X_test)
    base_metrics = compute_metrics(y_test, y_pred_base, p_features=len(feature_names))

    print(f"-> Baseline R2: {base_metrics['R2']:.4f}")
    print(f"-> Baseline RMSE: ${base_metrics['RMSE']:,.2f}")
    print(f"-> Baseline MAE: ${base_metrics['MAE']:,.2f}")

    # 4. Feature Selection
    print("\n[Step 5] 執行特徵選擇 (Feature Selection via Lasso)...")
    sel_idx, sel_feats, rem_feats, imp_df, fs_details = perform_feature_selection(
        X_train, y_train, feature_names, method="lasso", random_state=42
    )

    print(f"-> 原始特徵數量: {fs_details['original_count']}")
    print(f"-> 篩選方法: {fs_details['method']}")
    print(f"-> 選中的特徵數量: {len(sel_feats)}")
    print(f"-> 選中的特徵 (Selected): {', '.join(sel_feats)}")
    print(f"-> 被移除的特徵 (Removed): {', '.join(rem_feats)}")

    # 5. Model 2: Feature Selected Multiple Linear Regression
    print("\n[Step 6] 建立 Model 2: Feature Selected Multiple Linear Regression...")
    X_train_sel = X_train[:, sel_idx]
    X_test_sel = X_test[:, sel_idx]

    fs_model = LinearRegressionWithIntervals(fit_intercept=True)
    fs_model.fit(X_train_sel, y_train)

    y_pred_fs, ci_low_fs, ci_up_fs, pi_low_fs, pi_up_fs = fs_model.predict_intervals(X_test_sel)
    fs_metrics = compute_metrics(y_test, y_pred_fs, p_features=len(sel_feats))

    print(f"-> Feature Selected R2: {fs_metrics['R2']:.4f}")
    print(f"-> Feature Selected RMSE: ${fs_metrics['RMSE']:,.2f}")
    print(f"-> Feature Selected MAE: ${fs_metrics['MAE']:,.2f}")

    # 6. Evaluation & Comparison
    print("\n" + "=" * 70)
    print("  [Step 7] 模型評估與比較 (Model Evaluation Comparison)")
    print("=" * 70)
    comp_df = compare_models(base_metrics, fs_metrics)
    print(comp_df.to_string(index=False))

    # 7. Visualization & Plots
    print("\n[Step 8] 匯出預測圖表與殘差診斷圖 (Figures Export)...")
    os.makedirs("figures", exist_ok=True)

    # Actual vs Predicted with PI and CI
    fig1 = plot_actual_vs_predicted(
        y_test, y_pred_base, ci_low_base, ci_up_base, pi_low_base, pi_up_base,
        title="Baseline Model: Actual vs. Predicted with 95% CI & PI",
        save_path="figures/baseline_prediction_intervals.png"
    )

    fig2 = plot_actual_vs_predicted(
        y_test, y_pred_fs, ci_low_fs, ci_up_fs, pi_low_fs, pi_up_fs,
        title=f"Feature Selected Model ({len(sel_feats)} Features): Actual vs. Predicted with 95% CI & PI",
        save_path="figures/fs_prediction_intervals.png"
    )

    # Residuals
    fig3 = plot_residuals(y_test, y_pred_base, title_prefix="Baseline Model", save_path="figures/baseline_residuals.png")
    fig4 = plot_residuals(y_test, y_pred_fs, title_prefix="Feature Selected Model", save_path="figures/fs_residuals.png")

    print("-> 成功匯出圖表至 'figures/' 目錄:")
    print("   1. figures/baseline_prediction_intervals.png")
    print("   2. figures/fs_prediction_intervals.png")
    print("   3. figures/baseline_residuals.png")
    print("   4. figures/fs_residuals.png")

    print("\n" + "=" * 70)
    print("  Pipeline 執行完畢!")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
