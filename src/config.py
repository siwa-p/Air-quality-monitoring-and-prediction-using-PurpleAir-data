DB_PATH = "datasets/warehouse.duckdb"

NOAA_STATIONS = {
    "dallas":       "USW00003971",
    "philadelphia": "USW00013739",   # Philadelphia International Airport
}

BOUNDING_BOXES = {
    "dallas":       dict(nwlng=-97.43, nwlat=33.31, selng=-96.28, selat=32.37),
    "philadelphia": dict(nwlng=-75.65, nwlat=40.35, selng=-74.75, selat=39.65),
}

ACTIVE_CITY = "philadelphia"

TEST_PERIOD_START = "2024-01-08"
PM25_OUTLIER_THRESHOLD = 1000
