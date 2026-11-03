"""
Feature definitions and label helpers.

LEAKAGE GUARD: the model inputs are exactly config.FEATURE_COLUMNS - six academic
indicators that are known *before* the final examination. The final examination
marks are only ever used as the training target, never as an input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

FEATURES: list[str] = list(config.FEATURE_COLUMNS)
TARGET: str = config.TARGET_COLUMN

assert TARGET not in FEATURES, "Data leakage: the target must never be an input feature"


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select model inputs in a fixed order. Raises if any feature is missing."""
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[FEATURES].astype(float)


def risk_from_marks(marks: float) -> str:
    if marks >= config.LOW_RISK_MIN:
        return config.RISK_LOW
    if marks >= config.MODERATE_RISK_MIN:
        return config.RISK_MODERATE
    return config.RISK_HIGH


def risk_labels(marks: pd.Series | np.ndarray) -> np.ndarray:
    return np.array([risk_from_marks(float(m)) for m in marks])


def performance_category(marks: float) -> str:
    for lower, label in config.PERFORMANCE_BANDS:
        if marks >= lower:
            return label
    return config.PERFORMANCE_BANDS[-1][1]


def internal_pct(internal1: float, internal2: float) -> float:
    """Average of both internal tests as a percentage (each test is out of 50)."""
    return (float(internal1) + float(internal2)) / 100 * 100
