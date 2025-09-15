from __future__ import annotations
import io
from typing import Optional, Dict, Tuple
import streamlit as st
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import mapping
import rioxarray
import matplotlib.pyplot as plt
import folium
from folium.raster_layers import ImageOverlay

@st.cache_data
def mask_to_basin(da: xr.DataArray, basin_gdf: gpd.GeoDataFrame) -> xr.DataArray:
    """Clip/mask a 2D DataArray [lat, lon] to the basin polygon (assumes EPSG:4326)."""
    # rioxarray requires the CRS to be set on both the DataArray and the GeoDataFrame.
    # We can't be sure what the user has provided, so we'll standardize to EPSG:4326.
    if basin_gdf.crs is None:
        basin_gdf = basin_gdf.set_crs("EPSG:4326")
    else:
        basin_gdf = basin_gdf.to_crs("EPSG:4326")

    if not hasattr(da, "rio") or da.rio.crs is None:
        da = da.rio.write_crs("EPSG:4326", inplace=False)

    geom = [mapping(geom) for geom in basin_gdf.geometry]
    clipped = da.rio.clip(geom, basin_gdf.crs, drop=False)
    return clipped

@st.cache_data
def basin_mean(da: xr.DataArray, basin_gdf: gpd.GeoDataFrame) -> float:
    """Area-unweighted mean over masked pixels (nanmean)."""
    m = mask_to_basin(da, basin_gdf)
    return float(np.nanmean(m.values))

def render_raster_png(da: xr.DataArray, vmin: Optional[float]=None, vmax: Optional[float]=None) -> Tuple[bytes, Tuple[float,float,float,float]]:
    """Render a 2D DataArray to a PNG and return bytes + bounds (south, west, north, east)."""
    if "latitude" in da.dims and "longitude" in da.dims:
        lat = da["latitude"].values
        lon = da["longitude"].values
    elif "lat" in da.dims and "lon" in da.dims:
        lat = da["lat"].values
        lon = da["lon"].values
    else:
        raise ValueError("Expected dims to be latitude/longitude.")
    south, north = float(np.nanmin(lat)), float(np.nanmax(lat))
    west, east = float(np.nanmin(lon)), float(np.nanmax(lon))
    arr = da.values
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(arr, origin="lower", extent=[west, east, south, north], vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.read(), (south, west, north, east)

def folium_map_with_raster(da_masked: xr.DataArray, basin_gdf: gpd.GeoDataFrame, title: str) -> folium.Map:
    png_bytes, bounds = render_raster_png(da_masked)
    south, west, north, east = bounds
    centroid = basin_gdf.to_crs(4326).geometry.unary_union.centroid
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=8, tiles="CartoDB dark_matter")
    ImageOverlay(image=png_bytes, bounds=[[south, west],[north, east]], opacity=0.75, name=title).add_to(m)
    folium.GeoJson(basin_gdf.__geo_interface__, name="Basin").add_to(m)
    folium.LayerControl().add_to(m)
    return m

@st.cache_data
def combine_basin_boundaries(basins: Dict[str, str]) -> gpd.GeoDataFrame:
    """Given basin_name -> shapefile path, return a combined GeoDataFrame with a 'name' column."""
    import pandas as pd
    gdfs = []
    for name, shp in basins.items():
        try:
            gdf = gpd.read_file(shp)
            if gdf.crs is None:
                gdf.set_crs(4326, inplace=True)
            else:
                gdf = gdf.to_crs(4326)
            gdf["name"] = name
            gdf = gdf.dissolve(by="name", as_index=False)
            gdfs.append(gdf[["name","geometry"]])
        except Exception:
            continue
    if not gdfs:
        return gpd.GeoDataFrame(columns=["name","geometry"], geometry="geometry", crs=4326)
    out = pd.concat(gdfs, ignore_index=True)
    out.set_crs(4326, inplace=True)
    return out
