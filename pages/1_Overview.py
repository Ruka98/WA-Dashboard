import os
import re
import streamlit as st
from annotated_text import annotated_text
from rapidfuzz import process, fuzz
from utils.data_loader import find_basins
from utils.map_utils import combine_basin_boundaries
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Overview • WA Rapid Dashboard", page_icon="📄", layout="wide")

def pick_basin_name(names, query):
    if not query:
        return None
    match = process.extractOne(query, names, scorer=fuzz.WRatio)
    if match and match[1] >= 70:
        return match[0]
    return None

def render_annotated_overview(text: str):
    """Parse overview text and highlight keywords using annotated_text."""

    KEYWORDS = {
        "Precipitation": "red", "P": "red",
        "Evapotranspiration": "blue", "ETa": "blue",
        "Land Use": "green", "LU": "green",
        "Water Accounting": "orange", "WA": "orange",
        "Basin": "purple",
    }
    pattern = re.compile(f"({'|'.join(re.escape(k) for k in KEYWORDS)})", re.IGNORECASE)
    parts = pattern.split(text)

    output = []
    for part in parts:
        match = pattern.fullmatch(part)
        if match:
            # Find which keyword was matched (case-insensitive) to get the color
            key_matched = next(k for k in KEYWORDS if k.lower() == part.lower())
            color = KEYWORDS[key_matched]
            output.append((part, key_matched, color))
        else:
            output.append(part)
    annotated_text(*output)


def main():
    data_root = st.session_state.get("data_root")
    basins = find_basins(data_root)
    if not basins:
        st.warning("No basins detected. Set a valid data folder in the sidebar.")
        return

    st.markdown("## 🌍 Basin Selector & Overview")
    col1, col2 = st.columns([1,2], gap="large")

    with col1:
        names = list(basins.keys())
        query = st.text_input("Search basin", placeholder="Type basin name…")
        suggested = pick_basin_name(names, query)
        basin_name = st.selectbox("or pick from list", options=names, index=names.index(suggested) if suggested in names else 0)
        st.session_state["basin_name"] = basin_name

        st.markdown("### 📘 Overview")
        overview_txt_path = basins[basin_name].overview_txt
        if overview_txt_path and os.path.isfile(overview_txt_path):
            with open(overview_txt_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                render_annotated_overview(content)
        else:
            st.info("No `Overview.txt` found in this basin folder.")

    with col2:
        st.markdown("### 🗺️ Basin Boundaries (All)")
        shp_map = {n: bp.shapefile for n, bp in basins.items() if bp.shapefile}
        if not shp_map:
            st.info("No shapefiles available.")
            return
        gdf_all = combine_basin_boundaries(shp_map)
        if gdf_all.empty:
            st.info("Could not read shapefiles.")
            return
        centroid = gdf_all.to_crs(4326).geometry.unary_union.centroid
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=4, tiles="CartoDB dark_matter")
        folium.GeoJson(gdf_all.__geo_interface__, name="Basins",
                       tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["Basin:"])).add_to(m)
        st_folium(m, height=550, use_container_width=True)

if __name__ == "__main__":
    main()
