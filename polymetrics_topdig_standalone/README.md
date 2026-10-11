# Polymetrics with TopDiG boundary metrics

Drop-in updates to the supplied Polymetrics 0.2.2 source files. This integrates
`metrics_updated.py`'s PoLiS calculations into `polymetrics/metrics.py`, adds
TopDiG/Pix2Poly-style building-boundary pixel scores, and exposes both through
`evaluate()` and the command-line tool.

The original matching (`matching.py`), object-level precision/recall/F1,
mean object IoU, confidence-based mAP, shape statistics, and TP/FP/FN
annotation behavior remain in place. No road metrics are included.

## Standalone installation (keeps the original package unchanged)

This ZIP is a self-contained *development copy* with packaging metadata. It
should be extracted into a separate directory, not over your old package.
Run the following commands from the extracted directory. On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe -c "import polymetrics; print(polymetrics.__file__, polymetrics.__version__)"
```

On Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q tests
.venv/bin/python -c 'import polymetrics; print(polymetrics.__file__, polymetrics.__version__)'
```

Use the virtual environment's Python interpreter to run your own scripts and
the module's CLI. `-e` means that edits to the files in this directory take
effect immediately without reinstalling. Installing in `.venv` does not
change any old Python environment or its `polymetrics` installation.

The `[test]` extra installs pytest and OpenCV; for ordinary use of TopDiG
without pytest use `pip install -e '.[topo]'`. For original metrics only,
`pip install -e .` suffices. The version string `0.2.2+topdig` is a local
marker for this experimental fork, not an official published release.

## Python usage

```python
from polymetrics import evaluate

result = evaluate(
    "ground_truth.geojson",
    "predictions.geojson",
    iou_threshold=0.5,
    compute_map=True,
    topo_pixel_size=0.3,                # EPSG:3857 coordinate units per raster pixel
    topo_boundary_width=5,             # OpenCV stroke thickness in pixels
    topo_bounds=(xmin, ymin, xmax, ymax),  # full extent of one image/tile
)
print(result.topo_accuracy)     # PA^topo, all building boundaries
print(result.topo_f1)           # F1^topo, all building boundaries
print(result.topo_iou)          # IoU^topo, all building boundaries
print(result.matched_topo_iou)  # matched-pair diagnostic only
print(result.polis)             # mean PoLiS over matched polygons
```

**`topo_pixel_size` is required to enable boundary metrics.** Without it,
`topo_*` and `matched_topo_iou` are `None`, and the existing object-level
functions still run. A 5-pixel stroke is the Pix2Poly evaluation convention,
**not** a 5-pixel geometric buffer radius.

**CRS/pixel-resolution warning.** Current Polymetrics input handling always
reprojects geometries to EPSG:3857. Accordingly, `topo_pixel_size` and
`topo_bounds` must be supplied in those *reprojected* coordinate units, not
necessarily in meters on the ground or the original image CRS. A raster
pixel that represents 0.3 meters on the ground does **not** automatically
correspond to 0.3 EPSG:3857 coordinate units; Mercator scale distortion
and any raster reprojection must be considered. For rigorous reproduction,
transform the image's native pixel footprint into the Polymetrics evaluation
CRS consistently. The current implementation assumes a north-up raster
and square raster pixels in the evaluation CRS.

To obtain the exact pixel-grid correspondence with a raster in EPSG:3857,
use its true pixel size and `(left, bottom, right, top)` bounds.

## CLI usage

```bash
python -m polymetrics truth.geojson pred.geojson \
  --topo-pixel-size 0.3 \
  --topo-width 5 \
  --topo-bounds XMIN YMIN XMAX YMAX \
  --no-map --json
```

## Metric interpretation

TopDiG's topology-wise evaluation rasterizes building boundaries to a binary
mask, then computes ordinary pixel metrics. The paper calls its boundary
parameter a dilation factor of delta=5, but does not specify a complete
rasterization recipe. The public **Pix2Poly evaluator** uses
`cv2.polylines(..., thickness=5)` and is the concrete convention replicated
here. All rings (including holes), components of MultiPolygons, unmatched
objects, and false-positive buildings participate in the *tile-level*
`topo_accuracy`, `topo_f1`, and `topo_iou`. For both masks empty, the three
scores are 1.0 (the Pix2Poly convention).

The TopDiG/Pix2Poly papers report **mIoU^topo**, the average of these
foreground IoU values **over images**. The `Result.topo_iou` here is for one
call to `evaluate`, i.e., one combined tile or image. If evaluating multiple
images, compute the **arithmetic mean of each tile's `topo_iou`** to compare
against their reported mIoU; do not combine all tiles into one giant
geographic bounding box. Similarly average image-wise F1 and PA if
reproducing their reported evaluation.

`matched_topo_iou` reuses Polymetrics' Hungarian IoU match **unchanged** and
returns the mean boundary IoU of those pairs, evaluated on each pair's
locally inferred shared raster crop. It is **not** TopDiG mIoU^topo:
unmatched predictions and references are excluded. It is `None` when there
are no matched pairs. If you want a diagnostic of the geometry of detected
objects without conflating it with missed objects, use this alongside the
image-wise scores.

Unlike `topo_iou` and `topo_f1`, `topo_accuracy` includes true-negative pixels,
so it can be very high even for poor building boundaries. Always set
`topo_bounds` to the **full original image/tile extent** for meaningful PA.
If omitted, the code uses the joint bounding rectangle of all polygons plus
padding; accuracy then depends on the automatically selected crop.

The helper rasterizer avoids inventing artificial lines where polygons are
clipped by the image edge. Images/grids exceeding 40 million pixels raise a
clear exception; run tile-wise or choose a less fine evaluation resolution.

## Tests

```bash
python -m pytest -q tests
```

The included tests cover exact and offset shapes, unmatched FP/FN buildings,
empty sets, holes, MultiPolygons, image-extent dependence, and validation.

## Sources

- TopDiG (CVPR 2023), section 5.1 and 5.2: <https://openaccess.thecvf.com/content/CVPR2023/html/Yang_TopDiG_Class-Agnostic_Topological_Directional_Graph_Extraction_From_Remote_Sensing_Images_CVPR_2023_paper.html>
- Pix2Poly implementation of the TopDiG metric:
  <https://github.com/yeshwanth95/Pix2Poly/blob/master/eval/topdig_eval_utils/metrics/topdig_metrics.py>

These are boundary-similarity metrics; they do not, by themselves, directly
measure graph connectivity, intersection relationships, or building counts.
