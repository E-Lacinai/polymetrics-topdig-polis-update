"""Run with: python -m pytest -q tests/"""

import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box

from polymetrics import evaluate
from polymetrics.metrics import topology_scores
from polymetrics.shape import polygon_boundary_mask, topology_grid


def frame(*geoms):
    return gpd.GeoDataFrame(geometry=list(geoms), crs="EPSG:3857")


TILE = (0.0, 0.0, 100.0, 100.0)
A = box(20, 20, 40, 40)
B = box(60, 60, 80, 80)


def evaluate_topo(gt, pred, **kwargs):
    return evaluate(gt, pred, compute_map=False, topo_pixel_size=1.0,
                    topo_bounds=TILE, **kwargs)


def test_exact_match_scores_and_polis():
    r = evaluate_topo(frame(A), frame(A))
    assert (r.tp, r.fp, r.fn) == (1, 0, 0)
    assert r.mean_iou == 1.0
    assert r.polis == 0.0
    assert r.topo_accuracy == 1.0
    assert r.topo_f1 == 1.0
    assert r.topo_iou == 1.0
    assert r.matched_topo_iou == 1.0


def test_disjoint_buildings_are_not_matched_but_topo_penalizes():
    r = evaluate_topo(frame(A), frame(B))
    assert (r.tp, r.fp, r.fn) == (0, 1, 1)
    assert r.topo_f1 == 0.0
    assert r.topo_iou == 0.0
    assert 0 < r.topo_accuracy < 1
    assert r.matched_topo_iou is None


def test_extra_fp_penalized_in_tile_score_not_matched_diagnostic():
    r = evaluate_topo(frame(A), frame(A, B))
    assert (r.tp, r.fp, r.fn) == (1, 1, 0)
    assert 0 < r.topo_iou < 1
    assert 0 < r.topo_f1 < 1
    assert r.matched_topo_iou == 1.0
    assert r.polis == 0.0


def test_missing_building_penalizes_tile_score():
    r = evaluate_topo(frame(A, B), frame(A))
    assert (r.tp, r.fp, r.fn) == (1, 0, 1)
    assert 0 < r.topo_iou < 1
    assert r.matched_topo_iou == 1.0


def test_small_offset_decreases_topological_score():
    shifted = box(22, 20, 42, 40)
    r = evaluate_topo(frame(A), frame(shifted))
    assert r.tp == 1
    assert 0 < r.polis
    assert 0 < r.topo_iou < 1
    assert 0 < r.matched_topo_iou < 1


def test_multipolygon_and_holes_are_respected():
    polygon_with_hole = Polygon(
        [(10, 10), (90, 10), (90, 90), (10, 90)],
        holes=[[(40, 40), (60, 40), (60, 60), (40, 60)]],
    )
    filled = box(10, 10, 90, 90)
    r = evaluate_topo(frame(filled), frame(polygon_with_hole))
    assert 0 < r.topo_iou < 1  # hole ring is extra boundary
    r_multi = evaluate_topo(frame(A, B), frame(A.union(B)))
    assert r_multi.topo_iou == 1.0  # MultiPolygon draws both boundaries


def test_both_empty_convention_and_single_empty():
    both = evaluate_topo(frame(), frame())
    assert (both.topo_accuracy, both.topo_f1, both.topo_iou) == (1, 1, 1)
    assert both.matched_topo_iou is None
    empty_pred = evaluate_topo(frame(A), frame())
    assert empty_pred.topo_iou == 0
    assert empty_pred.topo_f1 == 0
    assert empty_pred.topo_accuracy < 1


def test_topology_opt_in_preserves_existing_api():
    r = evaluate(frame(A), frame(A), compute_map=False)
    assert r.topo_iou is None
    assert r.matched_topo_iou is None
    assert r.to_dict()["topo_accuracy"] is None
    assert r.polis == 0.0
    assert "annotated" not in r.to_dict()


def test_bounds_change_pixel_accuracy_but_not_boundary_iou():
    small = topology_scores(frame(A), frame(B), pixel_size=1,
                            bounds=(10, 10, 90, 90))
    large = topology_scores(frame(A), frame(B), pixel_size=1,
                            bounds=(-100, -100, 200, 200))
    assert small["topo_iou"] == large["topo_iou"] == 0
    assert large["topo_accuracy"] > small["topo_accuracy"]


def test_validation_and_raster_limit():
    with pytest.raises(ValueError, match="positive"):
        evaluate(frame(A), frame(A), compute_map=False, topo_pixel_size=0)
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_topo(frame(A), frame(A), topo_boundary_width=0)
    with pytest.raises(ValueError, match="pixels"):
        evaluate(frame(A), frame(A), compute_map=False,
                 topo_pixel_size=1, topo_bounds=(0, 0, 10000, 10000))
    with pytest.raises(ValueError, match="required"):
        evaluate(frame(A), frame(A), compute_map=False,
                 topo_bounds=TILE)


def test_raster_lines_do_not_create_false_tile_boundary_edges():
    # Only left part of building is in the crop; clipping should not
    # introduce a fake vertical boundary along the right-hand image edge.
    mask = polygon_boundary_mask(
        [box(10, 10, 60, 60)], bounds=(0, 0, 32, 64),
        shape=(64, 32), pixel_size=1, boundary_width=1,
    )
    assert mask.any()
    assert mask[:, -1].sum() < mask.sum() // 3


def test_grid_auto_bounds_and_non_integer_pixel_size():
    bounds, shape = topology_grid([A], [B], 0.3, boundary_width=5)
    assert shape[0] > 0 and shape[1] > 0
    assert bounds[0] < 20 and bounds[2] > 80
    s = topology_scores(frame(A), frame(A), pixel_size=0.3)
    assert s["topo_iou"] == 1
