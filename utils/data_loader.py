from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import xarray as xr
import geopandas as gpd

VAR_HINTS = {
    "P": ["P", "precip", "precipitation"],
    "ETa": ["ETa", "ET", "evapotranspiration"],
    "LU": ["LU", "landuse", "land_use", "landcover", "land_cover"],
}

def _pick_var(ds: xr.Dataset, hints: List[str]) -> Optional[str]:
    for h in hints:
        for v in ds.data_vars:
            if h.lower() in v.lower():
                return v
    # fallback: single variable dataset
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    return None

@dataclass
class BasinPaths:
    name: str
    root: str
    nc_P: Optional[str] = None
    nc_ETa: Optional[str] = None
    nc_LU: Optional[str] = None
    shapefile: Optional[str] = None
    yearly_csvs: Dict[int, str] = None
    yearly_pdfs: Dict[int, str] = None
    overview_txt: Optional[str] = None

def find_basins(data_root: str) -> Dict[str, BasinPaths]:
    """Scan a folder for basin subfolders and assemble paths."""
    basins: Dict[str, BasinPaths] = {}
    if not os.path.isdir(data_root):
        return basins

    for basin_name in sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]):
        broot = os.path.join(data_root, basin_name)
        nc_dir = os.path.join(broot, "NetCDF")
        shp_dir = os.path.join(broot, "Shapefile")
        res_dir = os.path.join(broot, "Results", "yearly")

        bp = BasinPaths(name=basin_name, root=broot, yearly_csvs={}, yearly_pdfs={})

        # NetCDFs
        if os.path.isdir(nc_dir):
            for f in os.listdir(nc_dir):
                if not f.lower().endswith(".nc"): 
                    continue
                path = os.path.join(nc_dir, f)
                try:
                    ds = xr.open_dataset(path)
                except Exception:
                    continue
                # detect candidate variables
                pvar = _pick_var(ds, VAR_HINTS["P"])
                etvar = _pick_var(ds, VAR_HINTS["ETa"])
                luvar = _pick_var(ds, VAR_HINTS["LU"])
                # decide which file is which by presence of var and temporal resolution length
                if pvar and pvar in ds and pvar.lower().startswith("p") and "time" in ds and len(ds["time"]) >= 12:
                    if bp.nc_P is None: 
                        bp.nc_P = path
                if etvar and etvar in ds and "time" in ds and len(ds["time"]) >= 12:
                    if bp.nc_ETa is None:
                        bp.nc_ETa = path
                if luvar and luvar in ds and "time" in ds and len(ds["time"]) <= 12:  # likely yearly
                    if bp.nc_LU is None:
                        bp.nc_LU = path

        # Shapefile
        if os.path.isdir(shp_dir):
            for f in os.listdir(shp_dir):
                if f.lower().endswith(".shp"):
                    bp.shapefile = os.path.join(shp_dir, f)
                    break

        # Yearly results
        if os.path.isdir(res_dir):
            for f in os.listdir(res_dir):
                lower = f.lower()
                path = os.path.join(res_dir, f)
                year_match = re.search(r"(20\d{2})", lower)
                if not year_match:
                    continue
                year = int(year_match.group(1))
                if lower.endswith(".csv"):
                    bp.yearly_csvs[year] = path
                elif lower.endswith(".pdf"):
                    bp.yearly_pdfs[year] = path

        # Overview
        for cand in ["Overview.txt", "overview.txt", "README.txt"]:
            p = os.path.join(broot, cand)
            if os.path.isfile(p):
                bp.overview_txt = p
                break

        if bp.nc_P or bp.nc_ETa or bp.nc_LU:
            basins[basin_name] = bp

    return basins

def open_ds(path: str) -> xr.Dataset:
    return xr.open_dataset(path)

def load_shapefile(path: str):
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf.set_crs(4326, inplace=True)
    else:
        gdf = gdf.to_crs(4326)
    return gdf

def parse_yearly_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # expect a single column "CLASS;SUBCLASS;VARIABLE;VALUE"
    if df.shape[1] == 1:
        parts = df.iloc[:,0].astype(str).str.split(";", expand=True)
        if parts.shape[1] >= 4:
            parts = parts.iloc[:, :4]
            parts.columns = ["CLASS", "SUBCLASS", "VARIABLE", "VALUE"]
            parts["VALUE"] = pd.to_numeric(parts["VALUE"], errors="coerce")
            return parts
    # fallback: try to coerce expected columns
    cols = [c.strip().upper() for c in df.columns]
    rename = {}
    for src, dst in [("CLASS","CLASS"),("SUBCLASS","SUBCLASS"),("VARIABLE","VARIABLE"),("VALUE","VALUE")]:
        if src in cols:
            rename[df.columns[cols.index(src)]] = dst
    out = df.rename(columns=rename)
    if "VALUE" in out.columns:
        out["VALUE"] = pd.to_numeric(out["VALUE"], errors="coerce")
    return out

def discover_vars(ds: xr.Dataset):
    out = {}
    for key, hints in VAR_HINTS.items():
        v = _pick_var(ds, hints)
        if v:
            out[key] = v
    return out
