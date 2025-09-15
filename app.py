import os
import streamlit as st
from utils.data_loader import find_basins

st.set_page_config(page_title="Water Accounting Rapid Dashboard", page_icon="💧", layout="wide")

st.markdown(
    "<h1 style='margin-top:0'>Water Accounting Rapid Dashboard</h1>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("⚙️ Settings")
    data_root = st.text_input(
        "Data folder (parent of basin folders)",
        value=os.path.abspath(os.path.join(os.getcwd(), "data", "Basin")),
        help="Point this to the folder that contains your basin subfolders."
    )
    st.session_state["data_root"] = data_root
    st.caption("Place your dataset at `./data/Basin` to use the default path.")

basins = find_basins(st.session_state["data_root"])
if not basins:
    st.warning("No basins found. Ensure the folder contains subfolders with NetCDF/Shapefile/Results.")
else:
    st.success(f"Detected **{len(basins)}** basin(s): " + ", ".join(basins.keys()))
    st.info("Use the pages on the left to explore Overview, Maps, Time-Series, and Yearly Reports.")
