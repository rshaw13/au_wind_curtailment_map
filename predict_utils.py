import pickle
from pathlib import Path
import numpy as np
import requests
import streamlit as st

openweather_API_KEY = st.secrets["openweather_API_KEY"]

def load_models(model_dir="app/models"):
    """
    Load all saved windfarm models from disk
    """
    models = {}
    model_dir = Path(model_dir)
    for pkl_file in model_dir.glob("*.pkl"):
        wf_name = pkl_file.stem
        with open(pkl_file, "rb") as f:
            models[wf_name] = pickle.load(f)
    return models

def fetch_openweather_windspeeds(duid_meta):
    """
    duid_meta: dict of duid -> metadata with lat/lon
    Returns: dict of duid -> windspeed (m/s)
    """
    ws_dict = {}
    for duid, meta in duid_meta.items():
        lat, lon = meta["lat"], meta["lon"]
        if lat is None or lon is None:
            ws_dict[duid] = None
            continue

        # Example API call (replace with your API key)
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={openweather_API_KEY}&units=metric"
        try:
            data = requests.get(url).json()
            ws_dict[duid] = data['wind']['speed']  # m/s
        except Exception:
            ws_dict[duid] = None
    return ws_dict

def predict_current_output(models, wind_speeds):
    """
    Predict current expected output for each windfarm.
    - models: dict of wf_name -> trained pipeline
    - wind_speeds: dict of wf_name -> current windspeed
    """
    predictions = {}
    for wf_name, model in models.items():
        ws = wind_speeds.get(wf_name)
        if ws is None:
            predictions[wf_name] = None
            continue
        y_pred = model.predict(np.array([[ws]]))[0]
        predictions[wf_name] = max(0, y_pred)  # non-negative
    return predictions