"""One-to-one bipartite matching between predictions and ground truth on IoU."""

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True, slots=True)
class Match:
    pred_idx: np.ndarray  # shape (k,), matched prediction row indices
    truth_idx: np.ndarray  # shape (k,), matched truth row indices
    iou: np.ndarray  # shape (k,), IoU per matched pair
    iou_matrix: np.ndarray  # shape (n_pred, n_truth), full IoU matrix (sparse via 0 fill)


def iou_matrix(preds: gpd.GeoDataFrame, truth: gpd.GeoDataFrame) -> np.ndarray:
    """Compute the dense pred-by-truth IoU matrix using an STRtree to skip non-overlapping pairs."""
    n_pred = len(preds)
    n_truth = len(truth)
    out = np.zeros((n_pred, n_truth), dtype=np.float64)
    if n_pred == 0 or n_truth == 0:
        return out
    pred_geoms = preds.geometry.values
    truth_geoms = truth.geometry.values
    pred_sindex = preds.sindex
    truth_areas = truth.geometry.area.to_numpy()
    pred_areas = preds.geometry.area.to_numpy()
    for t_idx in range(n_truth):
        t_geom = truth_geoms[t_idx]
        candidates = pred_sindex.query(t_geom, predicate="intersects")
        if len(candidates) == 0:
            continue
        for p_idx in candidates:
            inter = pred_geoms[p_idx].intersection(t_geom).area
            if inter == 0.0:
                continue
            union = pred_areas[p_idx] + truth_areas[t_idx] - inter
            if union > 0:
                out[p_idx, t_idx] = inter / union
    return out


def match(preds: gpd.GeoDataFrame, truth: gpd.GeoDataFrame, iou_threshold: float = 0.5) -> Match:
    """Hungarian bipartite assignment on -IoU; pairs below iou_threshold are dropped."""
    ious = iou_matrix(preds, truth)
    if ious.size == 0 or ious.max() == 0:
        empty = np.empty(0, dtype=np.int64)
        return Match(pred_idx=empty, truth_idx=empty, iou=np.empty(0), iou_matrix=ious)
    pred_rows, truth_cols = linear_sum_assignment(-ious)
    pair_iou = ious[pred_rows, truth_cols]
    keep = pair_iou >= iou_threshold
    return Match(
        pred_idx=pred_rows[keep],
        truth_idx=truth_cols[keep],
        iou=pair_iou[keep],
        iou_matrix=ious,
    )
