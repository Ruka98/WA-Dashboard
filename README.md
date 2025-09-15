# Water Accounting Rapid Dashboard

A modern **Streamlit** dashboard to explore basin-level Water Accounting datasets:
- **Monthly**: Precipitation (**P**), Evapotranspiration (**ETa**), **P−ET**
- **Yearly**: Land Use (LU), summary CSV + PDF reports
- **Maps**: Monthly rasters masked by basin polygons (shapefiles)
- **Charts**: Time-series, bars, pies for P / ETa / P−ET and LU composition
- **Overview**: Displays `Overview.txt` (if present) per basin

### Expected dataset structure
```
Basin/
  <BASIN NAME>/
    NetCDF/
      *_P_*.nc           # monthly precipitation (var name "P")
      *_ET*.nc           # monthly evapotranspiration (var name "ETa")
      *_LU_*.nc          # yearly landuse (var name "LU")
    Results/
      yearly/
        sheet1_2019.csv
        sheet1_2019.pdf
        ...
    Shapefile/
      <basin>.shp (+ .shx/.dbf/.prj/...)
    Overview.txt (optional)
```

> Variable names are auto-detected (case-insensitive contains: **P**, **ETa**, **LU**). Coordinates should be `time`, `latitude`, `longitude` in **EPSG:4326**.

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
```

Then set the **Data folder** in the sidebar to the parent path containing your basin folders (e.g., `./data/Basin`).
