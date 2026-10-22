"""
Preprocessing Module
Handles missing values, categorical encoding, feature scaling, and train-test split without data leakage.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(
    df: pd.DataFrame,
    target_col: str = "price",
    drop_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Optional[StandardScaler], Dict[str, Any]]:
    """
    Cleans and preprocesses dataframe for Multiple Linear Regression:
    1. Drop non-predictive columns (e.g. id, date)
    2. Handle missing values
    3. One-hot encode categorical variables
    4. Split into train and test sets
    5. Standardize features (StandardScaler)
    """
    df_clean = df.copy()

    # Default columns to ignore/drop if present
    if drop_cols is None:
        drop_cols = ["id", "ID", "Id", "date", "Date", "Unnamed: 0"]
    
    actual_drop = [c for c in drop_cols if c in df_clean.columns and c != target_col]
    if actual_drop:
        df_clean = df_clean.drop(columns=actual_drop)

    # Missing value handling
    missing_summary = df_clean.isnull().sum()
    total_missing = missing_summary.sum()

    # Drop rows where target is NaN
    df_clean = df_clean.dropna(subset=[target_col])

    # Separate target
    y = df_clean[target_col].values.astype(float)
    X_df = df_clean.drop(columns=[target_col])

    # Impute numerical features with median
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if X_df[col].isnull().any():
            median_val = X_df[col].median()
            X_df[col] = X_df[col].fillna(median_val)

    # Impute and One-Hot Encode categorical features
    cat_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in cat_cols:
        if X_df[col].isnull().any():
            mode_val = X_df[col].mode()[0]
            X_df[col] = X_df[col].fillna(mode_val)

    if cat_cols:
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)

    feature_names = X_df.columns.tolist()
    X = X_df.values.astype(float)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    prep_info = {
        "original_shape": df.shape,
        "processed_samples": len(df_clean),
        "dropped_columns": actual_drop,
        "total_missing_imputed": int(total_missing),
        "encoded_categorical": cat_cols,
        "final_feature_count": len(feature_names),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "target_col": target_col
    }

    return X_train, X_test, y_train, y_test, feature_names, scaler, prep_info
