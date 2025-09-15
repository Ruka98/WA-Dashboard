import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from utils.data_loader import find_basins, open_ds, load_shapefile, discover_vars
from utils.map_utils import mask_to_basin, basin_mean

st.set_page_config(page_title="Time Series • WA Rapid Dashboard", page_icon="📈", layout="wide")

def compute_series(dsP, dsE, shp_gdf):
    vP = discover_vars(dsP).get("P") or list(dsP.data_vars)[0]
    vE = discover_vars(dsE).get("ETa") or list(dsE.data_vars)[0]
    t = pd.to_datetime(dsP["time"].values)
    rows = []
    for i, ts in enumerate(t):
        daP = dsP[vP].isel(time=i)
        daE = dsE[vE].isel(time=i)
        p = basin_mean(daP, shp_gdf)
        e = basin_mean(daE, shp_gdf)
        rows.append({"time": ts, "P": p, "ETa": e, "P-ETa": p - e})
    df = pd.DataFrame(rows).set_index("time")
    return df

def compute_lu_composition(dsL, shp_gdf):
    vL = discover_vars(dsL).get("LU") or list(dsL.data_vars)[0]
    t = pd.to_datetime(dsL["time"].values)
    out = {}
    for i, ts in enumerate(t):
        da = dsL[vL].isel(time=i)
        da_mask = mask_to_basin(da, shp_gdf)
        vals = pd.Series(da_mask.values.ravel())
        vals = vals.replace([np.inf, -np.inf], np.nan).dropna().astype(int)
        counts = vals.value_counts().sort_index()
        out[ts.year] = counts
    df = pd.DataFrame(out).fillna(0).astype(int).T
    return df

def main():
    data_root = st.session_state.get("data_root")
    basins = find_basins(data_root)
    if not basins:
        st.warning("No basins detected. Set a valid data folder in the sidebar.")
        return

    basin_name = st.session_state.get("basin_name", list(basins.keys())[0])
    basin = basins[basin_name]

    st.markdown(f"## 📈 Time Series — **{basin_name}**")

    if not (basin.nc_P and basin.nc_ETa and basin.shapefile):
        st.error("Required datasets not available (need P, ETa NetCDFs and a basin shapefile).")
        return

    dsP = open_ds(basin.nc_P)
    dsE = open_ds(basin.nc_ETa)
    shp_gdf = load_shapefile(basin.shapefile)

    with st.spinner("Computing basin-average monthly series…"):
        df = compute_series(dsP, dsE, shp_gdf)

    st.subheader("Monthly Averages (Basin) — mm/month")
    fig1 = px.line(df.reset_index(), x="time", y=["P","ETa","P-ETa"], markers=True)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Yearly Aggregates (mean of months)")
    df_year = df.resample("Y").mean(numeric_only=True)
    df_year.index = df_year.index.year
    c1, c2 = st.columns(2)
    with c1:
        fig2 = px.bar(df_year.reset_index(), x="time", y=["P","ETa","P-ETa"], barmode="group")
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        fig3 = px.line(df_year.reset_index(), x="time", y=["P","ETa","P-ETa"], markers=True)
        st.plotly_chart(fig3, use_container_width=True)

    st.download_button("⬇️ Download monthly CSV", data=df.to_csv().encode("utf-8"),
                       file_name=f"{basin_name}_monthly_P_ETa.csv", mime="text/csv")

    if basin.nc_LU:
        st.subheader("Land Use Composition by Year (pixel counts)")
        dsL = open_ds(basin.nc_LU)
        df_lu = compute_lu_composition(dsL, shp_gdf)
        st.dataframe(df_lu)
        fig4 = px.bar(df_lu, barmode="stack", title="LU composition (counts)")
        st.plotly_chart(fig4, use_container_width=True)

if __name__ == "__main__":
    main()
