"""Aggregate metrics: precision, recall, F1, mean IoU, PoLiS, mAP@0.5, mAP@0.5:0.95."""

import warnings

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from polymetrics.matching import Match, iou_matrix, match

_MAP_THRESHOLDS = np.arange(0.5, 1.0, 0.05)
_PR_RECALL_GRID = np.linspace(0.0, 1.0, 101)


def primary_from_match(m, n_pred: int, n_truth: int) -> dict[str, float]:
    """precision, recall, F1, mean IoU from a pre-computed Match."""
    tp = len(m.pred_idx)
    fp = int(n_pred - tp)
    fn = int(n_truth - tp)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_iou = float(m.iou.mean()) if tp else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
    }


def primary(
    preds: gpd.GeoDataFrame,
    truth: gpd.GeoDataFrame,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Precision, recall, F1, mean IoU, and mean PoLiS at one IoU threshold."""
    # Compute the match once and reuse it for all object-level metrics.
    m = match(preds, truth, iou_threshold=iou_threshold)

    metrics = primary_from_match(m, len(preds), len(truth))
    metrics["polis"] = polis_from_match(m, preds, truth)

    return metrics


def _scores_from(preds: gpd.GeoDataFrame) -> np.ndarray | None:
    """Pull a confidence score column from prediction properties;
    tolerate `score` or `confidence`.
    """
    for key in ("score", "confidence"):
        if key in preds.columns:
            return preds[key].to_numpy(dtype=np.float64)
    return None


def _ap_from_sorted_ious(
    ious_sorted: np.ndarray,
    n_truth: int,
    iou_threshold: float,
) -> float:
    """101-point AP at one IoU threshold given an IoU matrix already sorted
    by descending score.

    Inner loop is O(M) per prediction via a boolean mask + argmax.
    """
    n_pred = ious_sorted.shape[0]

    tp = np.zeros(n_pred, dtype=np.float64)
    available = np.ones(n_truth, dtype=bool)

    masked = ious_sorted.copy()
    masked[masked < iou_threshold] = 0.0

    for i in range(n_pred):
        row = masked[i] * available
        j = int(np.argmax(row))

        if row[j] > 0.0:
            available[j] = False
            tp[i] = 1.0

    fp = 1.0 - tp

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / n_truth
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    precisions_envelope = np.maximum.accumulate(precisions[::-1])[::-1]

    interp = np.interp(
        _PR_RECALL_GRID,
        recalls,
        precisions_envelope,
        left=precisions_envelope[0],
        right=0.0,
    )

    return float(interp.mean())


def average_precision(
    ious: np.ndarray,
    scores: np.ndarray,
    n_truth: int,
    iou_threshold: float,
) -> float:
    """101-point AP at a single IoU threshold (COCO-style integration)."""
    if n_truth == 0 or len(scores) == 0:
        return 0.0

    order = np.argsort(-scores)

    return _ap_from_sorted_ious(
        ious[order],
        n_truth,
        iou_threshold,
    )


def map_scores_from(
    ious: np.ndarray,
    scores: np.ndarray | None,
    n_truth: int,
) -> dict[str, float | None]:
    """mAP from a pre-computed IoU matrix; reuses one sort + matrix
    across the 10 thresholds.
    """
    if scores is None:
        warnings.warn(
            "predictions carry no `score` or `confidence` property; "
            "mAP cannot be computed",
            stacklevel=2,
        )
        return {"map_50": None, "map_50_95": None}

    if n_truth == 0 or len(scores) == 0:
        return {"map_50": 0.0, "map_50_95": 0.0}

    ious_sorted = ious[np.argsort(-scores)]

    ap50 = _ap_from_sorted_ious(
        ious_sorted,
        n_truth,
        0.5,
    )

    ap_per_threshold = [
        _ap_from_sorted_ious(
            ious_sorted,
            n_truth,
            threshold,
        )
        for threshold in _MAP_THRESHOLDS
    ]

    return {
        "map_50": ap50,
        "map_50_95": float(np.mean(ap_per_threshold)),
    }


def map_scores(
    preds: gpd.GeoDataFrame,
    truth: gpd.GeoDataFrame,
) -> dict[str, float | None]:
    """mAP@0.5 and mAP@0.5:0.95 from raw inputs
    (builds the IoU matrix internally).
    """
    return map_scores_from(
        iou_matrix(preds, truth),
        _scores_from(preds),
        len(truth),
    )


def _polygon_vertices(geom) -> list[tuple[float, float]]:
    """Return non-closing exterior vertices of a Polygon.

    For MultiPolygon, vertices from all component polygons are returned.
    """
    if geom.is_empty:
        return []

    if geom.geom_type == "Polygon":
        return [(float(x), float(y)) for x, y, *_ in geom.exterior.coords[:-1]]

    if geom.geom_type == "MultiPolygon":
        vertices: list[tuple[float, float]] = []
        for polygon in geom.geoms:
            vertices.extend(
                (float(x), float(y)) for x, y, *_ in polygon.exterior.coords[:-1]
            )
        return vertices

    raise TypeError(
        f"PoLiS requires Polygon or MultiPolygon geometries, " f"got {geom.geom_type!r}"
    )


def _directed_polis(geom_a, geom_b) -> float:
    """Directed PoLiS distance from polygon A to polygon B.

    Computes the mean distance from every vertex of A to the nearest
    point on the boundary of B, as in Eq. (8) of Avbelj et al.
    """
    if geom_a.is_empty or geom_b.is_empty:
        raise ValueError("PoLiS is undefined for empty geometries")

    vertices = _polygon_vertices(geom_a)
    boundary = geom_b.boundary

    if not vertices:
        raise ValueError("PoLiS requires a polygon with exterior vertices")

    return float(np.mean([Point(vertex).distance(boundary) for vertex in vertices]))


def polygon_polis(geom_a, geom_b) -> float:
    """Symmetric PoLiS distance between two polygons.

    Lower is better; 0 means identical boundaries.

    The value has the same units as the coordinate system of the
    input geometries.
    """
    return 0.5 * (_directed_polis(geom_a, geom_b) + _directed_polis(geom_b, geom_a))


def polis_from_match(
    m: Match,
    preds: gpd.GeoDataFrame,
    truth: gpd.GeoDataFrame,
) -> float:
    """Mean PoLiS over the matched prediction/reference building pairs."""
    if len(m.pred_idx) == 0:
        return 0.0

    distances = [
        polygon_polis(
            preds.geometry.iloc[pred_idx],
            truth.geometry.iloc[truth_idx],
        )
        for pred_idx, truth_idx in zip(m.pred_idx, m.truth_idx)
    ]

    return float(np.mean(distances))


def polis(
    preds: gpd.GeoDataFrame,
    truth: gpd.GeoDataFrame,
    iou_threshold: float = 0.5,
) -> float:
    """Mean PoLiS distance over objects matched at the given IoU threshold.

    The IoU threshold is used only for establishing object correspondence.
    The PoLiS calculation itself is threshold-free.
    """
    m = match(preds, truth, iou_threshold=iou_threshold)
    return polis_from_match(m, preds, truth)


# TopDiG / Pix2Poly topology-wise metrics for building boundaries.
# These are pixel-level image/tile metrics, NOT object-detection precision/recall.


def topology_scores(
    preds: gpd.GeoDataFrame,
    truth: gpd.GeoDataFrame,
    *,
    pixel_size: float,
    boundary_width: int = 5,
    bounds: tuple[float, float, float, float] | None = None,
) -> dict[str, float]:
    """TopDiG-style PA^topo, F1^topo, and IoU^topo over ALL building outlines.

    Predicted and reference polygon boundaries are drawn as binary strokes on a
    *shared* grid. This includes FP and FN buildings, even if no object-level
    IoU match exists. The resulting confusion matrix counts foreground pixels.

    ``pixel_size`` is the projected coordinate distance per image pixel, not
    necessarily one meter. Set ``bounds`` to the complete image/tile extent
    for a meaningful PA^topo (which counts all background pixels). Without it,
    the union extent plus padding is used and PA depends on the crop.

    ``boundary_width=5`` matches Pix2Poly's OpenCV stroke thickness; it does
    not mean dilation by a 5-pixel radius. The paper itself describes a
    dilation parameter delta=5 but does not specify the exact rasterizer.

    For a single image, foreground IoU equals the quantity averaged across
    images to obtain mIoU^topo in TopDiG/Pix2Poly. No smoothing is applied.
    If both boundary masks are empty, all three scores are defined as 1.0.
    """
    from polymetrics import shape

    pred_geoms = preds.geometry
    truth_geoms = truth.geometry
    grid_bounds, grid_shape = shape.topology_grid(
        pred_geoms, truth_geoms, pixel_size, bounds, boundary_width
    )
    mask_pred = shape.polygon_boundary_mask(
        pred_geoms, bounds=grid_bounds, shape=grid_shape,
        pixel_size=pixel_size, boundary_width=boundary_width,
    )
    mask_gt = shape.polygon_boundary_mask(
        truth_geoms, bounds=grid_bounds, shape=grid_shape,
        pixel_size=pixel_size, boundary_width=boundary_width,
    )
    return _binary_topology_scores(mask_pred, mask_gt)


def _binary_topology_scores(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Compute binary pixel overlap metrics with explicit empty-set behavior."""
    tp = int(np.count_nonzero(pred & truth))
    pred_pos = int(np.count_nonzero(pred))
    truth_pos = int(np.count_nonzero(truth))
    fp = pred_pos - tp
    fn = truth_pos - tp
    union = tp + fp + fn
    if union == 0:
        # Pix2Poly's convention for an empty (void) image.
        return {"topo_accuracy": 1.0, "topo_f1": 1.0, "topo_iou": 1.0}
    f1_denom = 2 * tp + fp + fn
    return {
        "topo_accuracy": float((pred.size - fp - fn) / pred.size),
        "topo_f1": float(2 * tp / f1_denom) if f1_denom else 0.0,
        "topo_iou": float(tp / union),
    }


def matched_topology_iou_from_match(
    m: Match,
    preds: gpd.GeoDataFrame,
    truth: gpd.GeoDataFrame,
    *,
    pixel_size: float,
    boundary_width: int = 5,
) -> float | None:
    """Average boundary-pixel IoU on the package's existing IoU-matched pairs.

    This is an OPTIONAL object-level diagnostic; it is NOT TopDiG mIoU^topo.
    Unmatched buildings do not contribute, and the result is None if there
    are no matches. Matching is NOT recomputed for boundary metrics.
    """
    if len(m.pred_idx) == 0:
        return None
    scores = []
    for pred_idx, truth_idx in zip(m.pred_idx, m.truth_idx, strict=True):
        pred_one = preds.iloc[[int(pred_idx)]]
        truth_one = truth.iloc[[int(truth_idx)]]
        scores.append(topology_scores(
            pred_one, truth_one, pixel_size=pixel_size,
            boundary_width=boundary_width,
        )["topo_iou"])
    return float(np.mean(scores))
