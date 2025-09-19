
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import xarray as xr
import plotly.express as px
import plotly.graph_objects as go
import os, glob, pandas as pd, numpy as np
import geopandas as gpd
import fiona
import shapely
from shapely.geometry import shape as shp_shape, mapping
from shapely import wkb as shp_wkb

# --- File/Dir ---
BASE_DIR = os.getcwd()
BASIN_DIR = os.path.join(BASE_DIR, 'basins')

# ---------- File Finding Helpers ----------
def _first_existing(patterns):
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            hits.sort()
            return hits[-1]
    return None

def find_nc_file(basin_name, variable_type):
    netcdf_dir = os.path.join(BASIN_DIR, basin_name, 'NetCDF')
    if not os.path.isdir(netcdf_dir): return None
    if variable_type == 'P':
        pats = [os.path.join(netcdf_dir, '*_P_*.nc'), os.path.join(netcdf_dir, '*P*.nc')]
    elif variable_type == 'ET':
        pats = [os.path.join(netcdf_dir, '*_ETa_*.nc'), os.path.join(netcdf_dir, '*_ET_*.nc'), os.path.join(netcdf_dir, '*ET*.nc')]
    elif variable_type == 'LU':
        pats = [os.path.join(netcdf_dir, '*_LU_*.nc'), os.path.join(netcdf_dir, '*LandUse*.nc'), os.path.join(netcdf_dir, '*LU*.nc')]
    else:
        return None
    return _first_existing(pats)

def find_shp_file(basin_name):
    shp_dir = os.path.join(BASIN_DIR, basin_name, 'Shapefile')
    if not os.path.isdir(shp_dir): return None
    pats = [os.path.join(shp_dir, '*.shp')]
    return _first_existing(pats)

# ---------- Data Processing Helpers ----------
def _standardize_latlon(ds):
    lat_names = ['latitude', 'lat', 'y']
    lon_names = ['longitude', 'lon', 'x']
    lat = next((n for n in lat_names if n in ds.coords or n in ds.variables), None)
    lon = next((n for n in lon_names if n in ds.coords or n in ds.variables), None)
    if lat and lat != 'latitude': ds = ds.rename({lat: 'latitude'})
    if lon and lon != 'longitude': ds = ds.rename({lon: 'longitude'})
    return ds

def _pick_data_var(ds):
    exclude = {'time', 'latitude', 'longitude', 'crs', 'spatial_ref'}
    cands = [v for v in ds.data_vars if v not in exclude]
    if not cands: return None
    with_ll = [v for v in cands if {'latitude','longitude'}.issubset(set(ds[v].dims))]
    return with_ll[0] if with_ll else cands[0]

def _compute_mode(arr, axis=None):
    vals, counts = np.unique(arr, return_counts=True)
    if len(counts) == 0: return np.nan
    return vals[np.argmax(counts)]

def _coarsen_to_1km(da, is_categorical=False):
    if 'latitude' not in da.dims or 'longitude' not in da.dims: return da
    lat_vals, lon_vals = da['latitude'].values, da['longitude'].values
    lat_res = float(np.abs(np.diff(lat_vals)).mean()) if lat_vals.size > 1 else 0.009
    lon_res = float(np.abs(np.diff(lon_vals)).mean()) if lon_vals.size > 1 else 0.009

    target_deg = 1.0 / 111.0
    f_lat = max(1, int(round(target_deg / lat_res))) if lat_res > 0 else 1
    f_lon = max(1, int(round(target_deg / lon_res))) if lon_res > 0 else 1

    coarsen_dict = {'latitude': f_lat, 'longitude': f_lon}

    if is_categorical:
        return da.coarsen(coarsen_dict, boundary='trim').reduce(_compute_mode)
    else:
        return da.coarsen(coarsen_dict, boundary='trim').mean(skipna=True)

def load_and_process_data(basin_name, variable_type, year_start=None, year_end=None, aggregate_time=True):
    fp = find_nc_file(basin_name, variable_type)
    if not fp: return None, None, "NetCDF file not found"
    try:
        ds = xr.open_dataset(fp, decode_times=True)
        ds = _standardize_latlon(ds)
        var = _pick_data_var(ds)
        if not var: return None, None, "No suitable data variable in file"
        da = ds[var]

        # Time selection (year range)
        if 'time' in ds.coords and (year_start is not None or year_end is not None):
            ys = int(year_start) if year_start is not None else pd.to_datetime(ds['time'].values).min().year
            ye = int(year_end) if year_end is not None else pd.to_datetime(ds['time'].values).max().year
            target_start = pd.to_datetime(f"{ys}-01-01")
            target_end = pd.to_datetime(f"{ye}-12-31")
            da = da.sel(time=slice(target_start, target_end))

        # Aggregation
        if 'time' in da.dims and da.sizes.get('time', 0) > 1 and aggregate_time:
            if variable_type in ['P', 'ET']:
                da = da.sum(dim='time', skipna=True)
        elif 'time' in da.dims and not aggregate_time:
            pass
        elif 'time' in da.dims:
            da = da.isel(time=0)

        da_1km = _coarsen_to_1km(da, is_categorical=(variable_type == 'LU'))
        return da_1km, var, os.path.basename(fp)
    except Exception as e:
        return None, None, f"Error processing file: {e}"

# ---------- Shapefile Cleaning Helpers ----------
def _force_2d(geom):
    # strip Z if present
    try:
        return shp_wkb.loads(shp_wkb.dumps(geom, output_dimension=2))
    except Exception:
        return geom

def _repair_poly(geom):
    try:
        g = geom.buffer(0)
        return g if (g is not None and not g.is_empty) else geom
    except Exception:
        return geom

def load_all_basins_geodata():
    """
    Reads each basins/<BASIN>/Shapefile/*.shp into one GeoDataFrame (EPSG:4326).
    Handles MultiPolygons, Z coords, invalid geoms; reprojects from any CRS.
    """
    rows = []
    if not os.path.isdir(BASIN_DIR):
        return gpd.GeoDataFrame(columns=['basin', 'geometry'], geometry='geometry', crs='EPSG:4326')

    for b in sorted([d for d in os.listdir(BASIN_DIR) if os.path.isdir(os.path.join(BASIN_DIR, d))]):
        shp = find_shp_file(b)
        if not shp or not os.path.exists(shp):
            continue

        try:
            with fiona.open(shp) as src:
                crs_wkt = src.crs_wkt
                crs_obj = None
                if crs_wkt:
                    try:
                        crs_obj = gpd.GeoSeries([0], crs=crs_wkt).crs
                    except Exception:
                        crs_obj = None

                geoms = []
                for feat in src:
                    if not feat or not feat.get("geometry"):
                        continue
                    geom = shp_shape(feat["geometry"])
                    geom = _force_2d(geom)
                    geom = _repair_poly(geom)
                    if geom and not geom.is_empty and geom.geom_type in ("Polygon","MultiPolygon"):
                        geoms.append(geom)

                if not geoms:
                    continue

                gdf = gpd.GeoDataFrame({"basin": [b]*len(geoms)}, geometry=geoms, crs=crs_obj or "EPSG:4326")
                # reproject to EPSG:4326 for Mapbox
                try:
                    gdf = gdf.to_crs("EPSG:4326")
                except Exception:
                    gdf.set_crs("EPSG:4326", allow_override=True, inplace=True)

                # explode multiparts so Plotly gets simpler features
                try:
                    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
                except Exception:
                    gdf = gdf.explode().reset_index(drop=True)

                # tiny cleanup: drop empties again after explode
                gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]

                rows.append(gdf[["basin","geometry"]])

        except Exception as e:
            print(f"[WARN] Problem with {b}: {e}")
            continue

    if not rows:
        return gpd.GeoDataFrame(columns=['basin', 'geometry'], geometry='geometry', crs='EPSG:4326')

    all_gdf = gpd.GeoDataFrame(pd.concat(rows, ignore_index=True), geometry="geometry", crs="EPSG:4326")
    return all_gdf

def basins_geojson(gdf=None):
    gdf = ALL_BASINS_GDF if gdf is None else gdf
    if gdf is None or gdf.empty:
        return {"type": "FeatureCollection", "features": []}

    features = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        try:
            features.append({
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {"basin": row["basin"]}
            })
        except Exception as e:
            print(f"[WARN] Could not convert geometry for basin {row['basin']}: {e}")
            continue

    return {"type": "FeatureCollection", "features": features}

ALL_BASINS_GDF = load_all_basins_geodata()
print("Basins:", ALL_BASINS_GDF['basin'].nunique() if not ALL_BASINS_GDF.empty else 0,
      "| Features:", len(ALL_BASINS_GDF) if not ALL_BASINS_GDF.empty else 0,
      "| CRS:", ALL_BASINS_GDF.crs)

# ---------- Basin Map (Selection) ----------
def make_basin_selector_map(selected_basin=None):
    # subset to draw/highlight
    gdf = ALL_BASINS_GDF if (not selected_basin or selected_basin == 'all') else ALL_BASINS_GDF[ALL_BASINS_GDF['basin'] == selected_basin]

    if gdf is None or gdf.empty:
        fig = go.Figure()
        fig.update_layout(title="No basin shapefiles found.", xaxis={'visible': False}, yaxis={'visible': False})
        return fig

    # Build GeoJSON with "basin" in properties to use as feature id
    geojson = basins_geojson(gdf)

    # One entry per feature; all same z (single color)
    locations = [f["properties"]["basin"] for f in geojson["features"]]
    z_vals = [1] * len(locations)

    # Faint fill so the basin is visible, plus a strong outline
    ch = go.Choroplethmapbox(
        geojson=geojson,
        locations=locations,
        featureidkey="properties.basin",
        z=z_vals,
        colorscale=[[0, 'rgba(0, 102, 255, 0.18)'], [1, 'rgba(0, 102, 255, 0.18)']],  # faint blue fill
        marker=dict(line=dict(width=3 if selected_basin and selected_basin != 'all' else 1.8,
                              color='rgb(0, 90, 200)')),
        hovertemplate='%{location}<extra></extra>'
    )

    fig = go.Figure(ch)

    # --- compute center + zoom from bounds (EPSG:4326) ---
    minx, miny, maxx, maxy = gdf.total_bounds
    # padding
    pad_x = (maxx - minx) * 0.08 if maxx > minx else 0.1
    pad_y = (maxy - miny) * 0.08 if maxy > miny else 0.1
    west, east = float(minx - pad_x), float(maxx + pad_x)
    south, north = float(miny - pad_y), float(maxy + pad_y)

    center_lon = (west + east) / 2.0
    center_lat = (south + north) / 2.0
    span_lon = max(east - west, 0.001)
    span_lat = max(north - south, 0.001)

    import math
    map_w, map_h = 900.0, 600.0  # heuristic for zoom calc
    lon_zoom = math.log2(360.0 / (span_lon * 1.1)) + math.log2(map_w / 512.0)
    lat_zoom = math.log2(180.0 / (span_lat * 1.1)) + math.log2(map_h / 512.0)
    zoom = max(0.0, min(16.0, lon_zoom, lat_zoom))

    fig.update_layout(
        mapbox=dict(
            style='open-street-map',        # no token needed
            center=dict(lon=center_lon, lat=center_lat),
            zoom=zoom,
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        uirevision='keep'
    )
    return fig

# ---------- Plotting Helpers ----------
def add_shapefile_to_fig(fig, basin_name):
    """Adds a basin boundary outline to a Plotly figure (cartesian)."""
    shp_file = find_shp_file(basin_name)
    if shp_file and os.path.exists(shp_file):
        gdf = gpd.read_file(shp_file)
        try:
            gdf = gdf.to_crs("EPSG:4326")
        except Exception:
            # assume already in 4326
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        for geom in gdf.geometry:
            geom = _repair_poly(_force_2d(geom))
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == 'Polygon':
                x, y = geom.exterior.xy
                fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines',
                                         line=dict(color='black', width=2),
                                         name='Basin Boundary', showlegend=False))
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    x, y = poly.exterior.xy
                    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines',
                                             line=dict(color='black', width=2),
                                             name='Basin Boundary', showlegend=False))
    return fig

def create_empty_fig(message="No data to display"):
    fig = go.Figure()
    fig.update_layout(
        xaxis={'visible': False},
        yaxis={'visible': False},
        annotations=[{'text': message, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 16}}]
    )
    return fig

# ---------- Dash App ----------
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# --- Layout ---
basin_folders = [d for d in os.listdir(BASIN_DIR) if os.path.isdir(os.path.join(BASIN_DIR, d))] if os.path.isdir(BASIN_DIR) else []
basin_options = [{'label': 'View All', 'value': 'all'}] + [{'label': b, 'value': b} for b in sorted(basin_folders)]

app.layout = html.Div([
    html.H1("Basin Data Dashboard", style={'textAlign': 'center'}),

    # 1) Basin selection block (dropdown + map)
    html.Div([
        html.H3("1. Select Basin by name or by clicking the map"),
        dcc.Dropdown(id='basin-dropdown', options=basin_options,
                     value='all' if basin_folders else None, clearable=False),
        html.Div(dcc.Graph(id='basin-map', style={'height': '60vh'}), style={'marginTop': '10px'}),
        html.P(id='file-info-feedback', style={'fontSize': 12, 'color': '#666', 'marginTop': 10})
    ], style={'width': '90%', 'margin': 'auto', 'padding': '10px'}),

    html.Hr(),

    # 2) Land Use / Land Cover (single year)
    html.Div([
        html.H2("Land Use / Land Cover", style={'textAlign': 'center'}),
        html.Div([
            html.H4("Select Year", style={'textAlign': 'center'}),
            dcc.Dropdown(id='lu-year-dropdown', searchable=True, clearable=False),
        ], style={'width': '60%', 'margin': 'auto'}),
        dcc.Loading(dcc.Graph(id='lu-map-graph', style={'height': '70vh'}))
    ], className='section-container'),

    html.Hr(),

    # 3) Precipitation: year range
    html.Div([
        html.H2("Precipitation (P)", style={'textAlign': 'center'}),
        html.H4("Select Year Range", style={'textAlign': 'center'}),
        html.Div([
            html.Div([
                html.Label("Start Year"),
                dcc.Dropdown(id='p-start-year-dropdown', searchable=True, clearable=False),
            ], style={'width': '45%', 'display': 'inline-block'}),
            html.Div([
                html.Label("End Year"),
                dcc.Dropdown(id='p-end-year-dropdown', searchable=True, clearable=False),
            ], style={'width': '45%', 'display': 'inline-block'}),
        ], style={'width': '70%', 'margin': 'auto'}),
        html.Div([
            html.Div(dcc.Loading(dcc.Graph(id='p-map-graph')), style={'width': '50%', 'display': 'inline-block'}),
            html.Div(dcc.Loading(dcc.Graph(id='p-bar-graph')), style={'width': '50%', 'display': 'inline-block'})
        ], style={'height': '60vh'})
    ], className='section-container'),

    html.Hr(),

    # 4) Evapotranspiration: year range
    html.Div([
        html.H2("Evapotranspiration (ET)", style={'textAlign': 'center'}),
        html.H4("Select Year Range", style={'textAlign': 'center'}),
        html.Div([
            html.Div([
                html.Label("Start Year"),
                dcc.Dropdown(id='et-start-year-dropdown', searchable=True, clearable=False),
            ], style={'width': '45%', 'display': 'inline-block'}),
            html.Div([
                html.Label("End Year"),
                dcc.Dropdown(id='et-end-year-dropdown', searchable=True, clearable=False),
            ], style={'width': '45%', 'display': 'inline-block'}),
        ], style={'width': '70%', 'margin': 'auto'}),
        html.Div([
            html.Div(dcc.Loading(dcc.Graph(id='et-map-graph')), style={'width': '50%', 'display': 'inline-block'}),
            html.Div(dcc.Loading(dcc.Graph(id='et-bar-graph')), style={'width': '50%', 'display': 'inline-block'})
        ], style={'height': '60vh'})
    ], className='section-container'),

    html.Hr(),

    # 5) Water Balance (P - ET): year range
    html.Div([
        html.H2("Water Balance (P - ET)", style={'textAlign': 'center'}),
        html.H4("Select Year Range", style={'textAlign': 'center'}),
        html.Div([
            html.Div([
                html.Label("Start Year"),
                dcc.Dropdown(id='p-et-start-year-dropdown', searchable=True, clearable=False),
            ], style={'width': '45%', 'display': 'inline-block'}),
            html.Div([
                html.Label("End Year"),
                dcc.Dropdown(id='p-et-end-year-dropdown', searchable=True, clearable=False),
            ], style={'width': '45%', 'display': 'inline-block'}),
        ], style={'width': '70%', 'margin': 'auto'}),
        html.Div([
            html.Div(dcc.Loading(dcc.Graph(id='p-et-map-graph')), style={'width': '50%', 'display': 'inline-block'}),
            html.Div(dcc.Loading(dcc.Graph(id='p-et-bar-graph')), style={'width': '50%', 'display': 'inline-block'})
        ], style={'height': '60vh'})
    ], className='section-container'),
], style={'fontFamily': 'Arial, sans-serif'})

# --- Callbacks ---
@app.callback(
    Output('basin-map', 'figure'),
    [Input('basin-dropdown', 'value')]
)
def sync_map_with_dropdown(basin):
    # Update map highlight & zoom when dropdown changes
    return make_basin_selector_map(selected_basin=basin)

@app.callback(
    Output('basin-dropdown', 'value'),
    [Input('basin-map', 'clickData')],
    [State('basin-dropdown', 'value')]
)
def sync_dropdown_with_map(clickData, current_value):
    # When user clicks on map polygon, set dropdown accordingly
    if clickData and 'points' in clickData and clickData['points']:
        point = clickData['points'][0]
        # thanks to featureidkey/locations, 'location' is the basin name
        basin_name = point.get('location')
        if basin_name:
            return basin_name
    return current_value

@app.callback(
    [Output('lu-year-dropdown', 'options'),
     Output('lu-year-dropdown', 'value'),
     Output('p-start-year-dropdown', 'options'),
     Output('p-start-year-dropdown', 'value'),
     Output('p-end-year-dropdown', 'options'),
     Output('p-end-year-dropdown', 'value'),
     Output('et-start-year-dropdown', 'options'),
     Output('et-start-year-dropdown', 'value'),
     Output('et-end-year-dropdown', 'options'),
     Output('et-end-year-dropdown', 'value'),
     Output('p-et-start-year-dropdown', 'options'),
     Output('p-et-start-year-dropdown', 'value'),
     Output('p-et-end-year-dropdown', 'options'),
     Output('p-et-end-year-dropdown', 'value'),
     Output('file-info-feedback', 'children')],
    [Input('basin-dropdown', 'value')]
)
def init_controls(basin):
    if not basin or basin == 'all': 
        empty_options = []
        empty_value = None
        return [empty_options, empty_value] * 7 + ["All basins view - select a specific basin for details."]

    p_fp = find_nc_file(basin, 'P')
    et_fp = find_nc_file(basin, 'ET')
    lu_fp = find_nc_file(basin, 'LU')

    p_min_yr, p_max_yr = 1990, 2025
    et_min_yr, et_max_yr = 1990, 2025
    lu_min_yr, lu_max_yr = 1990, 2025

    if p_fp:
        with xr.open_dataset(p_fp) as ds:
            if 'time' in ds.coords and ds.sizes.get('time', 0) > 0:
                times = pd.to_datetime(ds['time'].values)
                p_min_yr = int(times.min().year)
                p_max_yr = int(times.max().year)
    if et_fp:
        with xr.open_dataset(et_fp) as ds:
            if 'time' in ds.coords and ds.sizes.get('time', 0) > 0:
                times = pd.to_datetime(ds['time'].values)
                et_min_yr = int(times.min().year)
                et_max_yr = int(times.max().year)
    if lu_fp:
        with xr.open_dataset(lu_fp) as ds:
            if 'time' in ds.coords and ds.sizes.get('time', 0) > 0:
                times = pd.to_datetime(ds['time'].values)
                lu_min_yr = int(times.min().year)
                lu_max_yr = int(times.max().year)

    p_et_min = max(p_min_yr, et_min_yr)
    p_et_max = min(p_max_yr, et_max_yr)
    if p_et_min > p_et_max:
        p_et_min, p_et_max = p_min_yr, p_max_yr

    def make_options(min_yr, max_yr):
        years = list(range(min_yr, max_yr + 1))
        return [{'label': str(y), 'value': y} for y in years]

    def make_default_values(years):
        if not years:
            return None, None
        start = years[-3] if len(years) > 2 else years[0]
        end = years[-1]
        return start, end

    lu_years = list(range(lu_min_yr, lu_max_yr + 1))
    lu_options = make_options(lu_min_yr, lu_max_yr)
    lu_value = lu_years[-1] if lu_years else None

    p_years = list(range(p_min_yr, p_max_yr + 1))
    p_options = make_options(p_min_yr, p_max_yr)
    p_start_value, p_end_value = make_default_values(p_years)

    et_years = list(range(et_min_yr, et_max_yr + 1))
    et_options = make_options(et_min_yr, et_max_yr)
    et_start_value, et_end_value = make_default_values(et_years)

    pet_years = list(range(p_et_min, p_et_max + 1))
    pet_options = make_options(p_et_min, p_et_max)
    pet_start_value, pet_end_value = make_default_values(pet_years)

    files_found = f"P file: {os.path.basename(p_fp) if p_fp else 'Not Found'} | ET file: {os.path.basename(et_fp) if et_fp else 'Not Found'} | LU file: {os.path.basename(lu_fp) if lu_fp else 'Not Found'}"

    return (
        lu_options, lu_value,
        p_options, p_start_value, p_options, p_end_value,
        et_options, et_start_value, et_options, et_end_value,
        pet_options, pet_start_value, pet_options, pet_end_value,
        files_found
    )

def create_hydrology_outputs(basin, start_year, end_year, vtype):
    if basin == 'all' or not basin:
        return create_empty_fig("Select a specific basin to view data."), create_empty_fig("Select a specific basin to view data.")

    if not (basin and start_year is not None and end_year is not None):
        return create_empty_fig("Missing selections."), create_empty_fig()

    ys, ye = int(start_year), int(end_year)
    if ys > ye:
        ys, ye = ye, ys

    da_ts, _, _ = load_and_process_data(basin, vtype, year_start=ys, year_end=ye, aggregate_time=False)
    if da_ts is None or (hasattr(da_ts, 'sizes') and da_ts.sizes.get('time', 0) == 0):
        return create_empty_fig(f"{vtype} data not available for {ys}-{ye}."), create_empty_fig()

    # Map: temporal sum across range
    da_map = da_ts.sum(dim='time', skipna=True)
    map_title = f"Total {vtype} ({ys}–{ye})"
    colorscale = 'Blues' if vtype == 'P' else 'YlOrRd'

    fig_map = px.imshow(da_map.values, x=da_map['longitude'], y=da_map['latitude'],
                        color_continuous_scale=colorscale, origin='lower', aspect='equal',
                        title=map_title, labels={'color': 'mm'})
    fig_map = add_shapefile_to_fig(fig_map, basin)

    # Bar chart: monthly mean across range (spatial mean first)
    spatial_dims = [d for d in ['latitude', 'longitude'] if d in da_ts.dims]
    spatial_mean_ts = da_ts.mean(dim=spatial_dims, skipna=True)
    with np.errstate(invalid='ignore'):
        monthly = spatial_mean_ts.groupby('time.month').mean(skipna=True).rename({'month': 'Month'})

    if hasattr(monthly, 'values') and np.isfinite(np.asarray(monthly.values)).any():
        month_names = [pd.to_datetime(m, format='%m').strftime('%b') for m in monthly['Month'].values]
        y_values = np.asarray(monthly.values).flatten()
        fig_bar = px.bar(x=month_names, y=y_values,
                         title=f"Mean Monthly {vtype} ({ys}–{ye})",
                         labels={'x': 'Month', 'y': f'Mean Daily {vtype} (mm)'} )
    else:
        fig_bar = create_empty_fig(f"No valid monthly data for {vtype} in {ys}–{ye}.")

    return fig_map, fig_bar

@app.callback(
    [Output('p-map-graph', 'figure'), Output('p-bar-graph', 'figure')],
    [Input('basin-dropdown', 'value'), Input('p-start-year-dropdown', 'value'), Input('p-end-year-dropdown', 'value')]
)
def update_p_outputs(basin, start_year, end_year):
    return create_hydrology_outputs(basin, start_year, end_year, 'P')

@app.callback(
    [Output('et-map-graph', 'figure'), Output('et-bar-graph', 'figure')],
    [Input('basin-dropdown', 'value'), Input('et-start-year-dropdown', 'value'), Input('et-end-year-dropdown', 'value')]
)
def update_et_outputs(basin, start_year, end_year):
    return create_hydrology_outputs(basin, start_year, end_year, 'ET')

@app.callback(
    Output('lu-map-graph', 'figure'),
    [Input('basin-dropdown', 'value'), Input('lu-year-dropdown', 'value')]
)
def update_lu_map(basin, year):
    if basin == 'all' or not basin:
        return create_empty_fig("Select a specific basin to view data.")

    if not (basin and year): return create_empty_fig("Select Basin and Year")

    da, _, _ = load_and_process_data(basin, 'LU', year_start=year, year_end=year)
    if da is None: return create_empty_fig(f"Land Use data not found for {year}")
    title = f"Land Use / Cover for {year}"
    classes = np.unique(da.values[np.isfinite(da.values)]).astype(int) if np.isfinite(da.values).any() else []

    fig = px.imshow(da.values, x=da['longitude'], y=da['latitude'],
                    color_continuous_scale='Viridis', origin='lower', aspect='equal', title=title)

    if len(classes) > 0:
        fig.update_coloraxes(colorbar=dict(tickmode='array', tickvals=classes, ticktext=[str(c) for c in classes]))
    fig = add_shapefile_to_fig(fig, basin)
    return fig

@app.callback(
    [Output('p-et-map-graph', 'figure'), Output('p-et-bar-graph', 'figure')],
    [Input('basin-dropdown', 'value'), Input('p-et-start-year-dropdown', 'value'), Input('p-et-end-year-dropdown', 'value')]
)
def update_p_et_outputs(basin, start_year, end_year):
    if basin == 'all' or not basin:
        return create_empty_fig("Select a specific basin to view data."), create_empty_fig("Select a specific basin to view data.")

    if not (basin and start_year is not None and end_year is not None):
        return create_empty_fig("Missing selections."), create_empty_fig()

    ys, ye = int(start_year), int(end_year)
    if ys > ye: ys, ye = ye, ys

    da_p_ts, _, _ = load_and_process_data(basin, 'P', year_start=ys, year_end=ye, aggregate_time=False)
    da_et_ts, _, _ = load_and_process_data(basin, 'ET', year_start=ys, year_end=ye, aggregate_time=False)

    if da_p_ts is None or da_et_ts is None:
        return create_empty_fig("P or ET data missing."), create_empty_fig()

    da_p_aligned, da_et_aligned = xr.align(da_p_ts, da_et_ts, join='inner')
    if da_p_aligned.sizes.get('time', 0) == 0:
        return create_empty_fig("No overlapping time steps for P and ET."), create_empty_fig()

    da_p_et_ts = da_p_aligned - da_et_aligned

    # Map: sum across range
    da_map = da_p_et_ts.sum(dim='time', skipna=True)
    map_title = f"Total Water Balance (P-ET) ({ys}–{ye})"

    fig_map = px.imshow(da_map.values, x=da_map['longitude'], y=da_map['latitude'],
                        color_continuous_scale='RdBu', origin='lower', aspect='equal',
                        title=map_title, labels={'color': 'mm'})
    fig_map = add_shapefile_to_fig(fig_map, basin)

    # Bar: monthly mean across range
    spatial_dims = [d for d in ['latitude', 'longitude'] if d in da_p_et_ts.dims]
    spatial_mean_p_et_ts = da_p_et_ts.mean(dim=spatial_dims, skipna=True)
    with np.errstate(invalid='ignore'):
        monthly = spatial_mean_p_et_ts.groupby('time.month').mean(skipna=True).rename({'month': 'Month'})

    if hasattr(monthly, 'values') and np.isfinite(np.asarray(monthly.values)).any():
        month_names = [pd.to_datetime(m, format='%m').strftime('%b') for m in monthly['Month'].values]
        y_values = np.asarray(monthly.values).flatten()
        fig_bar = px.bar(x=month_names, y=y_values,
                         title=f"Mean Monthly Water Balance (P-ET) ({ys}–{ye})",
                         labels={'x': 'Month', 'y': 'Mean Daily P-ET (mm)'} )
        fig_bar.update_traces(marker_color=['red' if v < 0 else 'blue' for v in y_values])
    else:
        fig_bar = create_empty_fig(f"No valid monthly data for P-ET in {ys}–{ye}.")

    return fig_map, fig_bar

# --- Run ---
if __name__ == '__main__':
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), debug=False)
