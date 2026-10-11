"""GeoJSON I/O. Reads via geopandas and reprojects to EPSG:3857."""

from pathlib import Path

import geopandas as gpd


def load(path: str | Path) -> gpd.GeoDataFrame:
    """Read a GeoJSON file and reproject to EPSG:3857. Treats CRS-less input as EPSG:4326."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    if gdf.crs.to_epsg() != 3857:
        gdf = gdf.to_crs(epsg=3857)
    return gdf
