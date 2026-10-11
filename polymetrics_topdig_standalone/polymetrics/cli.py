"""Argparse CLI: `polymetrics truth.geojson pred.geojson [options]`."""

import argparse
import json
import sys
from pathlib import Path

from polymetrics import evaluate


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="polymetrics",
        description="Object-level polygons + PoLiS + optional TopDiG boundary metrics.",
    )
    p.add_argument("truth", type=Path, help="ground-truth GeoJSON")
    p.add_argument("pred", type=Path, help="prediction GeoJSON")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for TP (default: 0.5)")
    p.add_argument("--no-map", action="store_true", help="skip mAP computation")
    p.add_argument("--topo-pixel-size", type=float, default=None,
                   help="enable TopDiG topo metrics; pixel size in EPSG:3857 units")
    p.add_argument("--topo-width", type=int, default=5,
                   help="OpenCV boundary-stroke thickness in pixels (default: 5)")
    p.add_argument("--topo-bounds", type=float, nargs=4,
                   metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                   help="complete raster tile bounds in EPSG:3857 (recommended for PA)")
    p.add_argument("--out", type=Path, help="write annotated GeoJSON (TP/FP/FN per feature) to this path")
    p.add_argument("--json", action="store_true", help="emit metrics as JSON (default: human-readable)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate(
        args.truth, args.pred,
        iou_threshold=args.iou, compute_map=not args.no_map,
        topo_pixel_size=args.topo_pixel_size,
        topo_boundary_width=args.topo_width,
        topo_bounds=tuple(args.topo_bounds) if args.topo_bounds is not None else None,
    )
    if args.out:
        result.to_geojson(args.out)
    if args.json:
        json.dump(result.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_human(result.to_dict())
    return 0


def _print_human(d: dict) -> None:
    rows = [
        ("IoU threshold", f"{d['iou_threshold']:.2f}"),
        ("TP / FP / FN", f"{d['tp']} / {d['fp']} / {d['fn']}"),
        ("Precision", f"{d['precision']:.4f}"),
        ("Recall", f"{d['recall']:.4f}"),
        ("F1", f"{d['f1']:.4f}"),
        ("Mean IoU (TP)", f"{d['mean_iou']:.4f}"),
        ("PoLiS (TP; projected units)", f"{d['polis']:.4f}"),
        ("mAP@0.5", f"{d['map_50']:.4f}" if d["map_50"] is not None else "n/a (no score property)"),
        ("mAP@0.5:0.95", f"{d['map_50_95']:.4f}" if d["map_50_95"] is not None else "n/a"),
        ("Avg vertices (pred)", f"{d['avg_vertices']:.2f}"),
        ("Orthogonality (pred)", f"{d['orthogonality']:.4f}"),
    ]
    if d["topo_iou"] is not None:
        rows.extend([
            ("PA-topo (tile pixels)", f"{d['topo_accuracy']:.4f}"),
            ("F1-topo (tile pixels)", f"{d['topo_f1']:.4f}"),
            ("IoU-topo (tile pixels)", f"{d['topo_iou']:.4f}"),
            ("Matched mean IoU-topo", f"{d['matched_topo_iou']:.4f}"
             if d['matched_topo_iou'] is not None else "n/a (no matches)"),
        ])
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {k:<{width}}  {v}")


if __name__ == "__main__":
    sys.exit(main())
