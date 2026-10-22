"""
Data Loader Module
Supports loading local CSV files and arbitrary remote CSV URLs with robust error handling.
"""

import pandas as pd
import numpy as np
import urllib.request
from typing import Tuple, Dict, Any, Optional


def validate_dataset(df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validates whether the dataframe is suitable for multiple linear regression.
    Checks:
    1. Not empty
    2. Minimum row count (>= 30)
    3. Minimum column count (>= 3)
    4. Target column presence and numerical type
    5. Sufficient numerical/convertible features
    """
    if df is None or df.empty:
        return False, "資料集為空或無法成功讀取。", {}

    n_rows, n_cols = df.shape
    if n_rows < 30:
        return False, f"資料筆數過少 ({n_rows} 筆)，進行多元線性迴歸建議至少 30 筆樣本。", {}

    if n_cols < 3:
        return False, f"特徵欄位過少 ({n_cols} 欄)，進行多元線性迴歸建議至少包含 2 個特徵與 1 個目標變數。", {}

    # If target is specified, check it
    if target_col and target_col in df.columns:
        if not np.issubdtype(df[target_col].dtype, np.number):
            return False, f"指定的目標欄位 '{target_col}' 不是數值型態 (dtype: {df[target_col].dtype})，無法進行多元線性迴歸。", {}
        
        # Check if target is binary / categorical with very few unique values
        unique_targets = df[target_col].nunique()
        if unique_targets <= 2:
            return False, f"指定的目標欄位 '{target_col}' 僅有 {unique_targets} 種唯一值，屬於分類任務 (Classification)，非連續型多元線性迴歸 (Regression)。", {}
    
    # Identify candidate numerical targets
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    candidate_targets = [c for c in numerical_cols if df[c].nunique() > 10]
    
    stats = {
        "rows": n_rows,
        "cols": n_cols,
        "missing_cells": int(df.isnull().sum().sum()),
        "missing_ratio": float((df.isnull().sum().sum() / (n_rows * n_cols)) * 100),
        "numerical_cols": numerical_cols,
        "categorical_cols": df.select_dtypes(exclude=[np.number]).columns.tolist(),
        "candidate_targets": candidate_targets
    }

    if len(numerical_cols) < 2:
        return False, f"數值型欄位不足 ({len(numerical_cols)} 欄)，無法進行多元線性迴歸。", stats

    return True, "資料集驗證通過，符合多元線性迴歸需求。", stats


def load_dataset(source: str, target_col: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], bool, str, Dict[str, Any]]:
    """
    Loads dataset from either a local file path or a remote URL.
    Returns:
        df: pd.DataFrame or None
        is_valid: bool
        message: str
        metadata: dict
    """
    source = source.strip()
    try:
        if source.startswith("http://") or source.startswith("https://"):
            # Set request header to prevent 403 on some raw endpoints
            req = urllib.request.Request(source, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                content_type = resp.headers.get("Content-Type", "")
                # Read into pandas
                df = pd.read_csv(source)
        else:
            df = pd.read_csv(source)
    except urllib.error.HTTPError as e:
        return None, False, f"HTTP 連線錯誤 (代碼 {e.code}): 無法從提供的 URL 下載資料集，請確認網址是否公開可存取。", {}
    except urllib.error.URLError as e:
        return None, False, f"網路連線異常: {e.reason}，請檢查輸入的 URL 格式。", {}
    except pd.errors.ParserError as e:
        return None, False, f"CSV 解析失敗: 該 URL 內容可能不是標準的 CSV 格式檔案 ({e})。", {}
    except FileNotFoundError:
        return None, False, f"找不到檔案: 本地路徑 '{source}' 不存在。", {}
    except Exception as e:
        return None, False, f"讀取資料集時發生非預期錯誤: {str(e)}", {}

    is_valid, msg, stats = validate_dataset(df, target_col=target_col)
    return df, is_valid, msg, stats
