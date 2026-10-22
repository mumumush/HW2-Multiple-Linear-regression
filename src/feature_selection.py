"""
Feature Selection Module
Provides LassoCV and Recursive Feature Elimination (RFE) for Multiple Linear Regression.
Explicitly tracks original features, selected features, and removed features.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.feature_selection import RFE


def perform_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    method: str = "lasso",
    n_features_to_select: Optional[int] = None,
    random_state: int = 42
) -> Tuple[np.ndarray, List[str], List[str], pd.DataFrame, Dict[str, Any]]:
    """
    Performs feature selection.
    
    Parameters:
        X_train: Scaled feature matrix of training set
        y_train: Target vector
        feature_names: List of all original feature names
        method: 'lasso' or 'rfe'
        n_features_to_select: Used for RFE (defaults to ~60% of features if None)
        random_state: Seed
        
    Returns:
        selected_indices: np.ndarray of chosen column indices
        selected_features: List of chosen feature names
        removed_features: List of dropped feature names
        importance_df: DataFrame with ranking/coefficients
        details: Metadata dictionary
    """
    n_original = len(feature_names)
    
    if method.lower() == "lasso":
        # Cross-validated Lasso to find optimal alpha
        lasso = LassoCV(cv=5, random_state=random_state, max_iter=5000)
        lasso.fit(X_train, y_train)
        
        coef = lasso.coef_
        # Non-zero coefficient features are selected
        selected_indices = np.where(np.abs(coef) > 1e-5)[0]
        
        # If all or too few selected due to high penalty, take top k features by absolute coefficient
        if len(selected_indices) == 0:
            top_k = max(3, n_original // 2)
            selected_indices = np.argsort(np.abs(coef))[-top_k:]
            
        selected_features = [feature_names[i] for i in selected_indices]
        removed_features = [feature_names[i] for i in range(n_original) if i not in selected_indices]
        
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": coef,
            "Abs_Coefficient": np.abs(coef),
            "Selected": [f in selected_features for f in feature_names]
        }).sort_values(by="Abs_Coefficient", ascending=False).reset_index(drop=True)
        
        details = {
            "method": "Lasso Regression (L1 Regularization)",
            "best_alpha": float(lasso.alpha_),
            "original_count": n_original,
            "selected_count": len(selected_features),
            "removed_count": len(removed_features),
        }
        
    elif method.lower() == "rfe":
        if n_features_to_select is None:
            n_features_to_select = max(3, int(n_original * 0.6))
        n_features_to_select = min(n_features_to_select, n_original - 1)
        
        base_lr = LinearRegression()
        rfe = RFE(estimator=base_lr, n_features_to_select=n_features_to_select, step=1)
        rfe.fit(X_train, y_train)
        
        selected_indices = np.where(rfe.support_)[0]
        selected_features = [feature_names[i] for i in selected_indices]
        removed_features = [feature_names[i] for i in range(n_original) if i not in selected_indices]
        
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Ranking": rfe.ranking_,
            "Selected": rfe.support_
        }).sort_values(by="Ranking", ascending=True).reset_index(drop=True)
        
        details = {
            "method": "Recursive Feature Elimination (RFE)",
            "n_features_target": n_features_to_select,
            "original_count": n_original,
            "selected_count": len(selected_features),
            "removed_count": len(removed_features),
        }
    else:
        raise ValueError(f"不支援的特徵選擇方法: {method}")

    return selected_indices, selected_features, removed_features, importance_df, details
