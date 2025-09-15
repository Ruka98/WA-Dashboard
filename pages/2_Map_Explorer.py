import os
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import xarray as xr
from utils.data_loader import find_basins, open_ds, load_shapefile, discover_vars
from utils.map_utils import mask_to_basin, folium_map_with_raster

st.set_page_config(page_title="Map Explorer • WA Rapid Dashboard", page_icon="🗺️", layout="wide")

def main():
    data_root = st.session_state.get("data_root")
    basins = find_basins(data_root)
    if not basins:
        st.warning("No basins detected. Set a valid data folder in the sidebar.")
        return

    basin_name = st.session_state.get("basin_name", list(basins.keys())[0])
    basin = basins[basin_name]

    st.markdown(f"## 🗺️ Map Explorer — **{basin_name}**")

    if not basin.shapefile:
        st.error("Shapefile not found for this basin.")
        return
    shp_gdf = load_shapefile(basin.shapefile)

    dsP = open_ds(basin.nc_P) if basin.nc_P else None
    dsE = open_ds(basin.nc_ETa) if basin.nc_ETa else None
    dsL = open_ds(basin.nc_LU) if basin.nc_LU else None

    colA, colB = st.columns([1,2], gap="large")
    with colA:
        var_choice = st.selectbox("Variable", ["P (mm/month)", "ETa (mm/month)", "P − ETa (mm/month)", "Land Use (yearly)"])
        if "Land Use" in var_choice:
            if dsL is None:
                st.error("Land Use dataset not available.")
                return
            t = pd.to_datetime(dsL["time"].values)
            year = st.selectbox("Year", sorted(list(set(t.year))), index=0)
            month = None
        else:
            if dsP is None or dsE is None:
                st.error("Monthly P and/or ETa dataset not available.")
                return
            t = pd.to_datetime(dsP["time"].values)
            years = sorted(list(set(t.year)))
            year = st.selectbox("Year", years, index=0)
            months = [m for m in range(1,13) if pd.Timestamp(year=year, month=m, day=1) in t]
            month = st.selectbox("Month", months, index=0)

    with colB:
        if "Land Use" in var_choice:
            var = discover_vars(dsL).get("LU") or list(dsL.data_vars)[0]
            idx = int(np.where(pd.to_datetime(dsL["time"].values).year == year)[0][0])
            da = dsL[var].isel(time=idx)
            da_mask = mask_to_basin(da, shp_gdf)
            title = f"LU — {year}"
        else:
            vP = discover_vars(dsP).get("P") or list(dsP.data_vars)[0]
            vE = discover_vars(dsE).get("ETa") or list(dsE.data_vars)[0]
            idx = int(np.where(pd.to_datetime(dsP["time"].values) == pd.Timestamp(year=year, month=month, day=1))[0][0])
            daP = dsP[vP].isel(time=idx)
            daE = dsE[vE].isel(time=idx)
            if "P − ETa" in var_choice:
                da = (daP - daE).rename("P_minus_ETa")
            elif "ETa" in var_choice:
                da = daE
            else:
                da = daP
            da_mask = mask_to_basin(da, shp_gdf)
            title = f"{da.name} — {year}-{month:02d}"

        fmap = folium_map_with_raster(da_mask, shp_gdf, title)
        st_folium(fmap, height=600, use_container_width=True)

        # Downloads
        import io, tempfile
        from utils.map_utils import render_raster_png
        png_bytes, _ = render_raster_png(da_mask)
        st.download_button("⬇️ Download PNG", data=png_bytes, file_name=f"{basin_name}_{title.replace(' ','_')}.png", mime="image/png")
        # GeoTIFF
        import rioxarray
        da_geo = da_mask.rio.write_crs(4326, inplace=False)
        with tempfile.TemporaryDirectory() as td:
            tif_path = os.path.join(td, f"{basin_name}_{title.replace(' ','_')}.tif")
            da_geo.rio.to_raster(tif_path)
            with open(tif_path, "rb") as f:
                st.download_button("⬇️ Download GeoTIFF", data=f.read(), file_name=os.path.basename(tif_path), mime="image/tiff")

if __name__ == "__main__":
    main()
