import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime
from map_utils import make_windfarm_map
from live_scraper import get_latest_scada_df, latest_output_dict
from predict_utils import load_models, predict_current_output
from anomaly_counter import update_anomaly_total

st.set_page_config(page_title="Australia Windfarm Live Monitor", layout="wide")

st.title("🌀 Australia Windfarm Live Performance Monitor")

# -------------------------
# Load metadata
# -------------------------
with open("app/metadata/windfarm_meta.json") as f:
    windfarm_meta = json.load(f)
with open("app/metadata/windfarm_duids.json") as f:
    windfarm_duids = json.load(f)

# -------------------------
# Load power curve models
# -------------------------
models = load_models("app/models")

# -------------------------
# Get latest SCADA output
# -------------------------
scada_df, source_file = get_latest_scada_df(
    remote_index_url="https://www.nemweb.com.au/REPORTS/CURRENT/Dispatch_SCADA/",
    local_zip_dir="app/saved_zips"
)

latest_actual = latest_output_dict(scada_df, wind_duids=windfarm_duids)

# -------------------------
# Windspeed Layer (OpenWeather Tile)
# For now no numerical windspeeds, only tile layer
# (Streamlit map overlays will use folium)
# -------------------------
wind_speeds = {}  # numerical values can be added later
wind_speeds = fetch_openweather_windspeeds(windfarm_meta)

# -------------------------
# Predict expected output
# -------------------------
predicted = predict_current_output(models, wind_speeds)

# -------------------------
# Build data table
# -------------------------
rows = []
for duid, meta in windfarm_meta.items():
    actual = latest_actual.get(duid)
    expected = predicted.get(duid)
    ws = wind_speeds.get(duid)

    if expected not in (None, 0) and actual not in (None,) and actual is not None:
        error_frac = max(0, (expected - actual) / expected)
    else:
        error_frac = None

rows.append({
    "DUID": duid,
    "Name": meta["name"],
    "Participant": meta["participant"],
    "Windspeed (m/s)": ws,
    "Actual (MW)": actual,
    "Expected (MW)": expected,
    "Error (%)": error_frac * 100 if error_frac is not None else None,
    "Lat": meta["lat"],
    "Lon": meta["lon"],
    "Capacity (MW)": meta["capacity_mw"],
})

df = pd.DataFrame(rows)

# -------------------------
# Update and show anomaly MW total
# -------------------------
total_deficit_mw = update_anomaly_total(df)

st.markdown(f"""
### ⚠️ Cumulative Anomalous MW  
Total MW lost due to underperformance so far:  
**{total_deficit_mw:.2f} MW**
""")

# -------------------------
# Map
# -------------------------
st.subheader("📍 Windfarm Performance Map")
map_html = make_windfarm_map(df)
st.components.v1.html(map_html, height=750)

# -------------------------
# Table
# -------------------------
st.subheader("Windfarm Output Table")
st.dataframe(df)

# -------------------------
# Auto-refresh every 15 min
# -------------------------
st.experimental_rerun()
time.sleep(900)  # 15 minutes