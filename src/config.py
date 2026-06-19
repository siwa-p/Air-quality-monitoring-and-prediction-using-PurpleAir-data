DB_PATH = "datasets/warehouse.duckdb"

NOAA_STATIONS = {
    "dallas":        "USW00003971",
    "philadelphia":  "USW00013739",   # Philadelphia International Airport
    "los_angeles":   "USW00023174",   # Los Angeles International Airport (LAX)
}

BOUNDING_BOXES = {
    "dallas":        dict(nwlng=-97.43, nwlat=33.31, selng=-96.28, selat=32.37),
    "philadelphia":  dict(nwlng=-75.65, nwlat=40.35, selng=-74.75, selat=39.65),
    "los_angeles":   dict(nwlng=-119.0, nwlat=34.8,  selng=-116.8, selat=33.4),
}

ACTIVE_CITY = "los_angeles"

# Tighter bbox for GNN/spatial models — dense LA basin core only.
# Excludes Lancaster, Victorville, Riverside (meteorologically distinct from basin).
SPATIAL_BBOX = dict(lon_min=-118.7, lat_min=33.7, lon_max=-117.8, lat_max=34.4)

DAILY_START = "2018-01-01"   # earliest date to fetch for daily sensor data
TEST_PERIOD_START = "2024-01-08"
PM25_OUTLIER_THRESHOLD = 200
