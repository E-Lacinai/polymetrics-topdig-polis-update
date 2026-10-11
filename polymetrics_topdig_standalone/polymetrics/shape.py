"""Polygon shape statistics and ground-truth-comparison deltas."""

import math
from itertools import chain

import geopandas as gpd
import numpy as np

_ORTHO_TOL_DEG = 5.0


def avg_vertex_count(gdf: gpd.GeoDataFrame) -> float:
    """Mean exterior vertex count across polygons (closing vertex excluded)."""
    if len(gdf) == 0:
        return 0.0
    counts = []
    for geom in gdf.geometry:
        if geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            counts.append(len(geom.exterior.coords) - 1)
        elif geom.geom_type == "MultiPolygon":
            counts.extend(len(p.exterior.coords) - 1 for p in geom.geoms)
    return float(np.mean(counts)) if counts else 0.0


def orthogonality(gdf: gpd.GeoDataFrame, tol_deg: float = _ORTHO_TOL_DEG) -> float:
    """Mean per-polygon fraction of exterior edges within `tol_deg` of the MBR dominant axis."""
    scores: list[float] = []
    for geom in gdf.geometry:
        if geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            scores.append(_polygon_orthogonality(geom, tol_deg))
        elif geom.geom_type == "MultiPolygon":
            scores.extend(_polygon_orthogonality(p, tol_deg) for p in geom.geoms)
    return float(np.mean(scores)) if scores else 0.0


def _polygon_orthogonality(polygon, tol_deg: float) -> float:
    """Fraction of edges within tol_deg of the dominant axis of the minimum rotated rectangle."""
    mbr = polygon.minimum_rotated_rectangle
    if mbr.geom_type != "Polygon" or mbr.is_empty:
        return 0.0
    mbr_coords = list(mbr.exterior.coords)
    dominant = _segment_angle(mbr_coords[0], mbr_coords[1])
    coords = list(polygon.exterior.coords)
    if len(coords) < 3:
        return 0.0
    aligned = 0
    total = 0
    for i in range(len(coords) - 1):
        angle = _segment_angle(coords[i], coords[i + 1])
        diff = abs(_angle_diff(angle, dominant)) % 90.0
        diff = min(diff, 90.0 - diff)
        if diff <= tol_deg:
            aligned += 1
        total += 1
    return aligned / total if total else 0.0


def compactness(gdf: gpd.GeoDataFrame) -> float:
    """Mean isoperimetric quotient (4*pi*A / P^2) across polygons. 1.0 = circle, 0.785 = square."""
    scores: list[float] = []
    for geom in gdf.geometry:
        if geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            score = _polygon_compactness(geom)
            if score is not None:
                scores.append(score)
        elif geom.geom_type == "MultiPolygon":
            for p in geom.geoms:
                score = _polygon_compactness(p)
                if score is not None:
                    scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def vertex_count_ratio(predictions: gpd.GeoDataFrame, truth: gpd.GeoDataFrame) -> float:
    """avg_vertex_count(predictions) / avg_vertex_count(truth). 1.0 = matches GT polygon complexity."""
    truth_count = avg_vertex_count(truth)
    if truth_count == 0.0:
        return 0.0
    return avg_vertex_count(predictions) / truth_count


def polygon_count_ratio(predictions: gpd.GeoDataFrame, truth: gpd.GeoDataFrame) -> float:
    """Count of pred features / count of truth features (empty geometries skipped).

    1.0 = same number of polygons as GT. <1.0 = under-segmentation (merged blocks).
    >1.0 = over-segmentation (fragmented instances). Targets watershed seed and area-filter knobs.
    """
    truth_n = sum(1 for g in truth.geometry if not g.is_empty)
    if truth_n == 0:
        return 0.0
    pred_n = sum(1 for g in predictions.geometry if not g.is_empty)
    return pred_n / truth_n


def orthogonality_delta(predictions: gpd.GeoDataFrame, truth: gpd.GeoDataFrame) -> float:
    """|orthogonality(predictions) - orthogonality(truth)|. 0.0 = matches GT rectilinearity profile."""
    return abs(orthogonality(predictions) - orthogonality(truth))


def compactness_delta(predictions: gpd.GeoDataFrame, truth: gpd.GeoDataFrame) -> float:
    """|compactness(predictions) - compactness(truth)|. 0.0 = matches GT shape regularity."""
    return abs(compactness(predictions) - compactness(truth))


def _polygon_compactness(polygon) -> float | None:
    area = polygon.area
    perimeter = polygon.length
    if perimeter <= 0 or area <= 0:
        return None
    return 4.0 * math.pi * area / (perimeter * perimeter)


def _segment_angle(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _angle_diff(a: float, b: float) -> float:
    diff = (a - b + 180.0) % 360.0 - 180.0
    return diff


# TopDiG-style boundary rasterization. Keeping this in shape.py separates
# polygon/raster geometry from the confusion-matrix calculations in metrics.py.


def _topology_lines(geom):
    """Yield constituent LineStrings from polygon boundary linework."""
    if geom.is_empty:
        return
    if geom.geom_type in ("LineString", "LinearRing"):
        yield geom
    elif geom.geom_type in ("MultiLineString", "GeometryCollection"):
        for part in geom.geoms:
            yield from _topology_lines(part)


def topology_grid(
    pred_geometries,
    truth_geometries,
    pixel_size: float,
    bounds: tuple[float, float, float, float] | None = None,
    boundary_width: int = 5,
    max_pixels: int = 40_000_000,
) -> tuple[tuple[float, float, float, float], tuple[int, int]]:
    """Create ONE shared raster grid for prediction/ground-truth boundaries.

    Coordinates are in the CRS used by Polymetrics (normally EPSG:3857).
    A supplied bounds tuple is the exact image footprint (xmin,ymin,xmax,ymax).
    If omitted, infer a padded extent from all polygons; pixel accuracy then
    describes this region of interest, not the full original image.

    The shape is (height, width). Pixel centers/rounding follow OpenCV's
    integer-coordinate polylines convention rather than an area rasterizer.
    """
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("topo_pixel_size must be a positive finite number")
    if not isinstance(boundary_width, (int, np.integer)) or boundary_width < 1:
        raise ValueError("topo_boundary_width must be a positive integer")
    if not isinstance(max_pixels, (int, np.integer)) or max_pixels < 1:
        raise ValueError("max_pixels must be a positive integer")

    if bounds is None:
        valid = []
        for geom in chain(pred_geometries, truth_geometries):
            if geom is not None and not geom.is_empty:
                valid.append(geom.bounds)
        if not valid:
            # Still allow the well-defined convention of an entirely empty tile.
            bounds = (0.0, 0.0, pixel_size, pixel_size)
        else:
            extent = np.asarray(valid, dtype=np.float64)
            padding = (boundary_width + 2) * pixel_size
            bounds = (
                float(extent[:, 0].min() - padding),
                float(extent[:, 1].min() - padding),
                float(extent[:, 2].max() + padding),
                float(extent[:, 3].max() + padding),
            )
    if len(bounds) != 4 or not np.all(np.isfinite(bounds)):
        raise ValueError("topo_bounds must be four finite numbers")
    xmin, ymin, xmax, ymax = map(float, bounds)
    if not xmin < xmax or not ymin < ymax:
        raise ValueError("topo_bounds must satisfy xmin < xmax and ymin < ymax")
    width = int(math.ceil((xmax - xmin) / pixel_size))
    height = int(math.ceil((ymax - ymin) / pixel_size))
    if width * height > max_pixels:
        raise ValueError(
            f"Topological raster would have {width * height:,} pixels; "
            "evaluate per tile or increase topo_pixel_size"
        )
    return (xmin, ymin, xmax, ymax), (height, width)


def polygon_boundary_mask(
    geometries,
    *,
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
    pixel_size: float,
    boundary_width: int = 5,
) -> np.ndarray:
    """Draw polygon boundaries (including hole rings) in a common pixel frame.

    Replicates Pix2Poly's rasterization convention: rounded pixel coordinates
    passed to cv2.polylines(thickness=5 by default). It is a stroke-width of
    five pixels, NOT a 5-pixel-radius geometric buffer. Each polygon is drawn
    on the same mask and overlaps are unioned. Outside-image edges are clipped
    without inventing an artificial polygon edge at the cropping boundary.

    Requires ``opencv-python-headless`` or ``opencv-python``.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "TopDiG metrics require opencv-python-headless or opencv-python"
        ) from exc
    from shapely.geometry import box

    height, width = shape
    xmin, ymin, xmax, ymax = bounds
    raster = np.zeros(shape, dtype=np.uint8)
    # Include strokes touching the frame by clipping to a slightly larger box.
    margin = (boundary_width + 2) * pixel_size
    clip = box(xmin - margin, ymin - margin, xmax + margin, ymax + margin)

    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type not in ("Polygon", "MultiPolygon"):
            raise TypeError(f"TopDiG boundaries need Polygon/MultiPolygon, got {geom.geom_type!r}")
        if not geom.intersects(clip):
            continue
        linework = geom.boundary.intersection(clip)
        for line in _topology_lines(linework):
            coords = np.asarray(line.coords, dtype=np.float64)
            if len(coords) < 2:
                continue
            # Polymetrics works in projected XY; numpy arrays use [row, column].
            pixels = np.rint(
                np.column_stack(((coords[:, 0] - xmin) / pixel_size,
                                 (ymax - coords[:, 1]) / pixel_size))
            )
            # cv2.polylines needs int32 and clips lines extending off the tile.
            points = np.clip(pixels, -(2**30), 2**30).astype(np.int32)
            cv2.polylines(raster, [points], isClosed=False,
                          color=1, thickness=int(boundary_width))
    return raster.astype(bool)
