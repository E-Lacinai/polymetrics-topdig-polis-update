"""Object-level (polygon) metrics for geospatial ML."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

__version__ = "0.2.2+topdig"

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from polymetrics import io, shape
from polymetrics.matching import Match, iou_matrix
from polymetrics.metrics import (
    _scores_from, map_scores_from, matched_topology_iou_from_match,
    polis_from_match, primary_from_match, topology_scores,
)

__all__ = ["Result", "evaluate"]


@dataclass(frozen=True, slots=True)
class Result:
    iou_threshold: float
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    mean_iou: float
    map_50: float | None
    map_50_95: float | None
    avg_vertices: float
    orthogonality: float
    annotated: gpd.GeoDataFrame = field(repr=False, compare=False)
    polis: float = 0.0
    topo_accuracy: float | None = None
    topo_f1: float | None = None
    topo_iou: float | None = None
    matched_topo_iou: float | None = None

    def to_dict(self) -> dict:
        out = asdict(self)
        del out["annotated"]
        return out

    def to_geojson(self, path: str | Path) -> Path:
        path = Path(path)
        self.annotated.to_file(path, driver="GeoJSON")
        return path


def evaluate(
    ground_truth: str | Path | gpd.GeoDataFrame,
    predictions: str | Path | gpd.GeoDataFrame,
    iou_threshold: float = 0.5,
    compute_map: bool = True,
    topo_pixel_size: float | None = None,
    topo_boundary_width: int = 5,
    topo_bounds: tuple[float, float, float, float] | None = None,
) -> Result:
    """End-to-end evaluation in EPSG:3857 (projected coordinates).

    Original object metrics and PoLiS use the SAME IoU/Hungarian matches.
    Supplying ``topo_pixel_size`` additionally enables TopDiG/Pix2Poly
    boundary-raster metrics. The three published topo scores evaluate all
    buildings on a shared image grid, including unmatched FP and FN objects.

    ``topo_pixel_size`` must be the original image pixel's size IN EPSG:3857
    coordinate units (often NOT 1). ``topo_bounds`` should specify the full
    tile footprint (xmin,ymin,xmax,ymax), especially for pixel accuracy.
    These metrics cannot be interpreted at a resolution-independent scale.
    """
    truth_gdf = _coerce(ground_truth)
    pred_gdf = _coerce(predictions)
    ious = iou_matrix(pred_gdf, truth_gdf)
    m = _match_from_iou(ious, iou_threshold)
    core = primary_from_match(m, len(pred_gdf), len(truth_gdf))
    ap = (
        map_scores_from(ious, _scores_from(pred_gdf), len(truth_gdf))
        if compute_map
        else {"map_50": None, "map_50_95": None}
    )
    if topo_pixel_size is None:
        if topo_bounds is not None or topo_boundary_width != 5:
            raise ValueError("topo_pixel_size is required when configuring topo evaluation")
        topo = {"topo_accuracy": None, "topo_f1": None, "topo_iou": None}
        matched_topo = None
    else:
        topo = topology_scores(
            pred_gdf, truth_gdf, pixel_size=topo_pixel_size,
            boundary_width=topo_boundary_width, bounds=topo_bounds,
        )
        matched_topo = matched_topology_iou_from_match(
            m, pred_gdf, truth_gdf, pixel_size=topo_pixel_size,
            boundary_width=topo_boundary_width,
        )
    annotated = _annotate(pred_gdf, truth_gdf, m, set(m.pred_idx.tolist()), set(m.truth_idx.tolist()))
    return Result(
        iou_threshold=iou_threshold,
        tp=int(core["tp"]),
        fp=int(core["fp"]),
        fn=int(core["fn"]),
        precision=float(core["precision"]),
        recall=float(core["recall"]),
        f1=float(core["f1"]),
        mean_iou=float(core["mean_iou"]),
        map_50=ap["map_50"],
        map_50_95=ap["map_50_95"],
        avg_vertices=shape.avg_vertex_count(pred_gdf),
        orthogonality=shape.orthogonality(pred_gdf),
        annotated=annotated,
        polis=polis_from_match(m, pred_gdf, truth_gdf),
        topo_accuracy=topo["topo_accuracy"],
        topo_f1=topo["topo_f1"],
        topo_iou=topo["topo_iou"],
        matched_topo_iou=matched_topo,
    )


def _match_from_iou(ious: np.ndarray, iou_threshold: float) -> Match:
    """Hungarian on -IoU using a pre-computed matrix; pairs below threshold are dropped."""
    if ious.size == 0 or ious.max() == 0:
        empty = np.empty(0, dtype=np.int64)
        return Match(pred_idx=empty, truth_idx=empty, iou=np.empty(0), iou_matrix=ious)
    pred_rows, truth_cols = linear_sum_assignment(-ious)
    pair_iou = ious[pred_rows, truth_cols]
    keep = pair_iou >= iou_threshold
    return Match(pred_idx=pred_rows[keep], truth_idx=truth_cols[keep], iou=pair_iou[keep], iou_matrix=ious)


def _coerce(src: str | Path | gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if isinstance(src, gpd.GeoDataFrame):
        if src.crs is None:
            src = src.set_crs(epsg=4326)
        if src.crs.to_epsg() != 3857:
            src = src.to_crs(epsg=3857)
        return gpd.GeoDataFrame(src.reset_index(drop=True), crs=src.crs)
    return io.load(src)


def _annotate(
    preds: gpd.GeoDataFrame,
    truth: gpd.GeoDataFrame,
    m: Match,
    tp_pred: set[int],
    matched_truth: set[int],
) -> gpd.GeoDataFrame:
    pred_iou_map = dict(zip(m.pred_idx.tolist(), m.iou.tolist(), strict=True))
    pred_rows = preds.copy()
    pred_rows["pm_class"] = ["TP" if i in tp_pred else "FP" for i in range(len(pred_rows))]
    pred_rows["pm_iou"] = [pred_iou_map.get(i, 0.0) for i in range(len(pred_rows))]
    pred_rows["pm_kind"] = "prediction"
    fn_mask = [i not in matched_truth for i in range(len(truth))]
    fn_rows = truth.iloc[fn_mask].copy()
    fn_rows["pm_class"] = "FN"
    fn_rows["pm_iou"] = 0.0
    fn_rows["pm_kind"] = "truth"
    return gpd.GeoDataFrame(pd.concat([pred_rows, fn_rows], ignore_index=True), crs=preds.crs)
