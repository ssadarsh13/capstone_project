"""Feature and prediction drift detection.

Methods
-------
- Numeric features : Population Stability Index (PSI) over quantile bins,
  plus a two-sample Kolmogorov–Smirnov test.
- Categorical / prediction distributions : PSI over categories, plus a
  chi-square test of independence.

PSI bands (industry convention): <0.10 stable, 0.10–0.25 moderate, >=0.25 significant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from .config import KS_PVALUE_ALPHA, psi_band

_EPS = 1e-6


@dataclass
class DriftResult:
    feature: str
    kind: str            # "numeric" | "categorical"
    psi: float
    psi_band: str
    stat_test: str       # "ks" | "chi2"
    statistic: float
    p_value: float
    drifted: bool        # PSI significant OR test significant

    def to_dict(self) -> Dict[str, object]:
        return {
            "feature": self.feature,
            "kind": self.kind,
            "psi": round(self.psi, 4),
            "psi_band": self.psi_band,
            "stat_test": self.stat_test,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "drifted": self.drifted,
        }


def _psi_numeric(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """PSI using quantile bin edges derived from the reference distribution."""
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        return 0.0
    # Quantile edges; fall back to linear if reference is near-constant.
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, bins=edges)[0] / ref.size
    cur_pct = np.histogram(cur, bins=edges)[0] / cur.size
    ref_pct = np.clip(ref_pct, _EPS, None)
    cur_pct = np.clip(cur_pct, _EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref.dropna().unique()) | set(cur.dropna().unique()))
    ref_pct = (ref.value_counts(normalize=True).reindex(cats).fillna(0)).to_numpy()
    cur_pct = (cur.value_counts(normalize=True).reindex(cats).fillna(0)).to_numpy()
    ref_pct = np.clip(ref_pct, _EPS, None)
    cur_pct = np.clip(cur_pct, _EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def numeric_drift(name: str, ref: pd.Series, cur: pd.Series) -> DriftResult:
    r = pd.to_numeric(ref, errors="coerce").to_numpy(dtype=float)
    c = pd.to_numeric(cur, errors="coerce").to_numpy(dtype=float)
    psi = _psi_numeric(r, c)
    r_clean, c_clean = r[~np.isnan(r)], c[~np.isnan(c)]
    if r_clean.size and c_clean.size:
        ks_stat, p = stats.ks_2samp(r_clean, c_clean)
    else:
        ks_stat, p = 0.0, 1.0
    band = psi_band(psi)
    drifted = bool(band == "significant" or p < KS_PVALUE_ALPHA)
    return DriftResult(name, "numeric", psi, band, "ks", float(ks_stat), float(p), drifted)


def categorical_drift(name: str, ref: pd.Series, cur: pd.Series) -> DriftResult:
    psi = _psi_categorical(ref.astype(str), cur.astype(str))
    cats = sorted(set(ref.dropna().astype(str)) | set(cur.dropna().astype(str)))
    ref_counts = ref.astype(str).value_counts().reindex(cats).fillna(0)
    cur_counts = cur.astype(str).value_counts().reindex(cats).fillna(0)
    table = np.vstack([ref_counts.to_numpy(), cur_counts.to_numpy()])
    table = table[:, table.sum(axis=0) > 0]  # drop empty categories
    if table.shape[1] > 1:
        chi2, p, _, _ = stats.chi2_contingency(table)
    else:
        chi2, p = 0.0, 1.0
    band = psi_band(psi)
    drifted = bool(band == "significant" or p < KS_PVALUE_ALPHA)
    return DriftResult(name, "categorical", psi, band, "chi2", float(chi2), float(p), drifted)


def compute_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
) -> List[DriftResult]:
    results: List[DriftResult] = []
    for col in numeric_features:
        if col in reference.columns and col in current.columns:
            results.append(numeric_drift(col, reference[col], current[col]))
    for col in categorical_features:
        if col in reference.columns and col in current.columns:
            results.append(categorical_drift(col, reference[col], current[col]))
    # Most drifted first.
    return sorted(results, key=lambda r: r.psi, reverse=True)


def compute_prediction_drift(
    ref_predictions: pd.Series, cur_predictions: pd.Series, name: str = "prediction"
) -> DriftResult:
    """Drift in the model's *output* class distribution (label shift signal)."""
    return categorical_drift(name, ref_predictions, cur_predictions)
