import os
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import find_basins, parse_yearly_csv

st.set_page_config(page_title="Yearly Reports • WA Rapid Dashboard", page_icon="📊", layout="wide")

def main():
    data_root = st.session_state.get("data_root")
    basins = find_basins(data_root)
    if not basins:
        st.warning("No basins detected. Set a valid data folder in the sidebar.")
        return

    basin_name = st.session_state.get("basin_name", list(basins.keys())[0])
    basin = basins[basin_name]

    st.markdown(f"## 📊 Yearly Reports — **{basin_name}**")

    years = sorted(set(list(basin.yearly_csvs.keys()) + list(basin.yearly_pdfs.keys())))
    if not years:
        st.info("No yearly CSV/PDF files found under `Results/yearly/`.")
        return

    year = st.selectbox("Year", years, index=0)

    csv_path = basin.yearly_csvs.get(year)
    pdf_path = basin.yearly_pdfs.get(year)

    if csv_path and os.path.isfile(csv_path):
        df = parse_yearly_csv(csv_path)
        st.subheader("Summary Table")
        st.dataframe(df)

        st.subheader("Stacked Bar by CLASS/SUBCLASS (sum of VALUE)")
        pivot = df.pivot_table(index="SUBCLASS", columns="CLASS", values="VALUE", aggfunc="sum").fillna(0)
        fig = px.bar(pivot, barmode="stack")
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("⬇️ Download CSV (clean)", data=df.to_csv(index=False).encode("utf-8"),
                           file_name=f"{basin_name}_summary_{year}.csv", mime="text/csv")
    else:
        st.info("No CSV for this year.")

    st.subheader("PDF Report")
    if pdf_path and os.path.isfile(pdf_path):
        st.markdown(f"[Open yearly PDF]({pdf_path})")
        st.caption("PDFs open in a new tab/window depending on your browser.")
    else:
        st.info("No PDF for this year.")

if __name__ == "__main__":
    main()
