"""Preprocessing and feature engineering utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

RAW_NUMERIC = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
]

class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip numeric columns to training-set IQR bounds; robust to future outliers."""
    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.columns_ = list(X.columns)
        self.lower_ = {}
        self.upper_ = {}
        for c in self.columns_:
            if pd.api.types.is_numeric_dtype(X[c]):
                q1, q3 = X[c].quantile([0.25, 0.75])
                iqr = q3 - q1
                self.lower_[c] = q1 - self.factor * iqr
                self.upper_[c] = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.columns_:
            if c in self.lower_:
                X[c] = X[c].clip(self.lower_[c], self.upper_[c])
        return X


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create physically meaningful features without target leakage."""
    x = data.copy()
    x["Temperature difference [K]"] = x["Process temperature [K]"] - x["Air temperature [K]"]
    x["Power [W]"] = x["Torque [Nm]"] * x["Rotational speed [rpm]"] * 2 * np.pi / 60
    x["Torque-speed interaction"] = x["Torque [Nm]"] * x["Rotational speed [rpm]"]
    x["Tool wear x torque"] = x["Tool wear [min]"] * x["Torque [Nm]"]
    x["Tool wear ratio"] = x["Tool wear [min]"] / 241.0
    x["High tool wear"] = (x["Tool wear [min]"] >= 180).astype(int)
    x["High temperature gap risk"] = (x["Temperature difference [K]"] < 8.6).astype(int)
    drop = ["UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
    return x.drop(columns=[c for c in drop if c in x.columns])
