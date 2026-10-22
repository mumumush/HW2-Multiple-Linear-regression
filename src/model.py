"""
Model Module: Multiple Linear Regression with Rigorous CI and PI
Implements OLS Multiple Linear Regression and computes exact statistical
Confidence Intervals (CI) and Prediction Intervals (PI).
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from scipy import stats
from sklearn.linear_model import LinearRegression


class LinearRegressionWithIntervals:
    """
    Multiple Linear Regression wrapper that provides:
    - OLS estimation
    - Standard errors, t-statistics, p-values
    - Exact Confidence Intervals (CI) for mean response
    - Exact Prediction Intervals (PI) for individual predictions
    """
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.model = LinearRegression(fit_intercept=fit_intercept)
        self.coef_ = None
        self.intercept_ = None
        self.n_samples = 0
        self.n_features = 0
        self.df_resid = 0
        self.mse_resid = 0.0
        self.rmse_resid = 0.0
        self.xtx_inv = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits OLS and calculates covariance matrix of parameters."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        self.n_samples, self.n_features = X.shape
        self.model.fit(X, y)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_ if self.fit_intercept else 0.0

        # Construct design matrix with intercept column for variance calculation
        if self.fit_intercept:
            X_design = np.hstack([np.ones((self.n_samples, 1)), X])
            p = self.n_features + 1
        else:
            X_design = X
            p = self.n_features

        self.df_resid = max(1, self.n_samples - p)

        # Residuals and variance
        y_pred_train = self.model.predict(X)
        residuals = y - y_pred_train
        sse = np.sum(residuals ** 2)
        self.mse_resid = sse / self.df_resid
        self.rmse_resid = np.sqrt(self.mse_resid)

        # Invert (X^T X) using pseudo-inverse for numerical stability
        try:
            self.xtx_inv = np.linalg.pinv(X_design.T @ X_design)
        except Exception:
            self.xtx_inv = np.linalg.inv(X_design.T @ X_design + 1e-6 * np.eye(X_design.shape[1]))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point prediction y_hat."""
        return self.model.predict(np.asarray(X, dtype=float))

    def predict_intervals(
        self,
        X: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates exact Confidence Interval (CI) and Prediction Interval (PI).
        
        Confidence Interval for Mean Response:
            y_hat +/- t_{alpha/2, df} * s * sqrt(h_00)
            
        Prediction Interval for Individual Observation:
            y_hat +/- t_{alpha/2, df} * s * sqrt(1 + h_00)
            
        Returns:
            y_pred: Point prediction
            ci_lower: Confidence interval lower bound
            ci_upper: Confidence interval upper bound
            pi_lower: Prediction interval lower bound
            pi_upper: Prediction interval upper bound
        """
        X = np.asarray(X, dtype=float)
        y_pred = self.predict(X)
        
        if self.fit_intercept:
            X_design = np.hstack([np.ones((len(X), 1)), X])
        else:
            X_design = X

        # Leverage h_00 = x_0 (X^T X)^-1 x_0^T for each test point
        # Efficient vectorized diagonal computation: sum((X_design @ xtx_inv) * X_design, axis=1)
        leverages = np.sum((X_design @ self.xtx_inv) * X_design, axis=1)
        leverages = np.clip(leverages, 0.0, None)  # Prevent small numerical negatives

        # Critical t-value
        t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=self.df_resid)

        # Standard error of the mean response
        se_mean = self.rmse_resid * np.sqrt(leverages)
        ci_lower = y_pred - t_crit * se_mean
        ci_upper = y_pred + t_crit * se_mean

        # Standard error of individual prediction
        se_pred = self.rmse_resid * np.sqrt(1.0 + leverages)
        pi_lower = y_pred - t_crit * se_pred
        pi_upper = y_pred + t_crit * se_pred

        return y_pred, ci_lower, ci_upper, pi_lower, pi_upper
