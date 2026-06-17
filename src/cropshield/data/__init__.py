"""
Data ingestion modules for CropShield.

Submodules
----------
fetch_nass
    Fetches county-level yield data from USDA NASS Quick Stats API.
fetch_power
    Fetches daily weather data from NASA POWER API by county centroid.
fetch_drought_monitor
    Fetches weekly drought severity data from the U.S. Drought Monitor.
build_county_panel
    Merges all interim data sources into the final modeling panel.
"""
