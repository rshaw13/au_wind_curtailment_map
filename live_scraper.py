"""
live_scraper.py

Functions:
- find_latest_remote_zip(index_url) -> str or None
- download_file(url, out_path)
- unzip_and_find_csv(zip_path) -> path_of_csv_in_zip
- load_scada_from_csv(csv_path) -> pd.DataFrame with columns ['SETTLEMENTDATE','DUID','SCADAVALUE']
- pick_latest_local_zip(dir_path) -> Path or None
- get_latest_scada_df(remote_index_url=None, local_zip_dir=None, download_dir='tmp_downloads') -> DataFrame
- build_windfarm_meta(reg_csv_path) -> dict of metadata keyed by DUID/name
- latest_output_dict(scada_df, wind_duids=None, agg='last') -> dict {DUID: latest_value}
- build_map_payload(models, latest_output, windspeed_dict, windfarm_meta, threshold=0.25) -> dict ready for /anomaly and /windfarms
"""

import re
import requests
import zipfile
import io
import os
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

# --------------------------
# CONFIG (edit as needed)
# --------------------------
DEFAULT_INDEX_URL = "https://www.nemweb.com.au/REPORTS/CURRENT/Dispatch_SCADA/"
DOWNLOAD_DIR = Path("tmp_downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)
REGISTRATION_CSV = Path("Full NEM Plant Registration List.csv")  # adjust path
# --------------------------

# Helper: find latest remote zip (by datetime in filename)
def find_latest_remote_zip(index_url=DEFAULT_INDEX_URL, pattern=r".*Dispatch_SCADA.*?(\d{8}_\d{6}).*\.zip"):
    """
    Scrape index_url and find the URL of the latest ZIP file, matching datetime pattern.
    Returns full URL string or None if not found.
    """
    try:
        resp = requests.get(index_url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        candidates = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            m = re.search(pattern, href)
            if m:
                dt_str = m.group(1)
                # parse: many filenames are like YYYYMMDD_HHMMSS
                try:
                    dt = datetime.strptime(dt_str, "%Y%m%d_%H%M%S")
                except ValueError:
                    # try alternative format
                    try:
                        dt = datetime.strptime(dt_str, "%Y%m%d")
                    except Exception:
                        continue
                # build full url
                full_url = href if href.startswith("http") else requests.compat.urljoin(index_url, href)
                candidates.append((dt, full_url))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    except Exception as e:
        print("Error finding remote zip:", e)
        return None

# Helper: pick latest local zip by filename datetime (same pattern)
def pick_latest_local_zip(local_dir, pattern=r".*Dispatch_SCADA.*?(\d{8}_\d{6}).*\.zip"):
    local_dir = Path(local_dir)
    if not local_dir.exists():
        return None
    candidates = []
    for f in local_dir.iterdir():
        if not f.is_file():
            continue
        m = re.search(pattern, f.name)
        if m:
            try:
                dt = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
            except ValueError:
                try:
                    dt = datetime.strptime(m.group(1), "%Y%m%d")
                except Exception:
                    continue
            candidates.append((dt, f))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

# Download a file to disk
def download_file(url, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
    return out_path

# Unzip and find the CSV inside (returns first CSV path)
def unzip_and_find_csv(zip_path, extract_to=None):
    zip_path = Path(zip_path)
    extract_to = Path(extract_to or zip_path.parent)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
        # find CSVs extracted
        for name in z.namelist():
            if name.lower().endswith(".csv"):
                return extract_to / name
    return None

# Load SCADA CSV into DataFrame with proper columns
def load_scada_from_csv(csv_path):
    """
    Loads the SCADA CSV (the zipped file contains the CSV). Returns a DataFrame
    with columns: SETTLEMENTDATE (datetime), DUID (string), SCADAVALUE (float)
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    # Defensive: try to find expected columns
    cols = [c.lower() for c in df.columns]
    # Map columns ignoring case/spacing
    col_map = {}
    for c in df.columns:
        key = c.strip().lower().replace(" ", "").replace("_", "")
        if "settlementdate" in key or "datetime" in key:
            col_map["SETTLEMENTDATE"] = c
        elif "duid" == key or key.endswith("duid"):
            col_map["DUID"] = c
        elif "scadavalue" in key or "scadavalue" in key:
            col_map["SCADAVALUE"] = c
    # fallback guesses
    if "SETTLEMENTDATE" not in col_map:
        possible = [c for c in df.columns if "date" in c.lower() and "time" in c.lower() or "settlement" in c.lower()]
        if possible:
            col_map["SETTLEMENTDATE"] = possible[0]
    if "DUID" not in col_map:
        possible = [c for c in df.columns if c.upper().strip() in ("DUID","UNIT","GENERATOR")]
        if possible:
            col_map["DUID"] = possible[0]
    if "SCADAVALUE" not in col_map:
        possible = [c for c in df.columns if "value" in c.lower() and ("scada" in c.lower() or "dispatch" in c.lower())]
        if possible:
            col_map["SCADAVALUE"] = possible[0]

    if not all(k in col_map for k in ("SETTLEMENTDATE","DUID","SCADAVALUE")):
        raise ValueError(f"Could not map columns in {csv_path.name}. Found: {df.columns.tolist()}")

    df = df[[col_map["SETTLEMENTDATE"], col_map["DUID"], col_map["SCADAVALUE"]]].copy()
    df.columns = ["SETTLEMENTDATE", "DUID", "SCADAVALUE"]
    # parse datetime
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"], errors="coerce")
    df = df.dropna(subset=["SETTLEMENTDATE"])
    # convert SCADAVALUE numeric
    df["SCADAVALUE"] = pd.to_numeric(df["SCADAVALUE"], errors="coerce")
    return df

# High-level: get latest scada df (remote attempt, else local)
def get_latest_scada_df(remote_index_url=None, local_zip_dir=None, download_dir=DOWNLOAD_DIR):
    """
    Returns (scada_df, source_path) where scada_df is a dataframe (SETTLEMENTDATE,DUID,SCADAVALUE)
    """
    remote_index_url = remote_index_url or DEFAULT_INDEX_URL
    # 1) try remote
    zip_url = find_latest_remote_zip(remote_index_url)
    if zip_url:
        try:
            print("Found remote zip:", zip_url)
            filename = zip_url.split("/")[-1]
            out_path = Path(download_dir) / filename
            download_file(zip_url, out_path)
            csv_path = unzip_and_find_csv(out_path, extract_to=out_path.parent)
            df = load_scada_from_csv(csv_path)
            return df, out_path
        except Exception as e:
            print("Remote fetch failed:", e)
    # 2) fallback to local directory with saved zips
    if local_zip_dir:
        candidate = pick_latest_local_zip(local_zip_dir)
        if candidate:
            try:
                print("Using local zip:", candidate)
                csv_path = unzip_and_find_csv(candidate, extract_to=candidate.parent)
                df = load_scada_from_csv(csv_path)
                return df, candidate
            except Exception as e:
                print("Loading local zip failed:", e)
    # 3) fallback: try to find any CSVs in local_zip_dir or current working dir
    # search for any CSV with 'Dispatch' or 'SCADA' in name
    search_dirs = [Path(local_zip_dir) if local_zip_dir else Path("."), Path(".")]
    for sd in search_dirs:
        for f in sd.rglob("*.csv"):
            if "dispatch" in f.name.lower() or "scada" in f.name.lower():
                try:
                    df = load_scada_from_csv(f)
                    return df, f
                except Exception:
                    continue
    raise FileNotFoundError("No SCADA CSV found remotely or locally.")

# Build windfarm metadata from registration CSV
def build_windfarm_meta(registration_csv_path=REGISTRATION_CSV, duid_col='DUID', lat_col_candidates=None, lon_col_candidates=None, capacity_col_candidates=None):
    """
    Reads Full NEM Plant Registration List.csv and returns a dict keyed by DUID (or plant name)
    containing lat/lon/capacity and display name.
    """
    df = pd.read_csv(registration_csv_path, low_memory=False)
    cols = [c.lower() for c in df.columns]

    # find duid-like column
    duid_col_name = None
    for c in df.columns:
        if c.strip().lower() == 'duid' or 'duid' in c.strip().lower():
            duid_col_name = c
            break
    if duid_col_name is None:
        raise ValueError("Could not find DUID column in registration csv")

    # find lat / lon
    lat_col = None
    lon_col = None
    for c in df.columns:
        lc = c.lower()
        if 'lat' in lc and lat_col is None:
            lat_col = c
        if ('lon' in lc or 'long' in lc) and lon_col is None:
            lon_col = c

    # find capacity
    cap_col = None
    for c in df.columns:
        if 'capacity' in c.lower() or 'capacity(mw)' in c.lower() or 'capacitymw' in c.lower():
            cap_col = c
            break
    # fallback name column
    name_col = None
    for c in df.columns:
        if 'name' in c.lower() or 'plant' in c.lower():
            name_col = c
            break

    meta = {}
    for _, row in df.iterrows():
        duid = str(row[duid_col_name]).strip()
        if not duid or duid.lower() in ('nan','none'):
            continue
        lat = float(row[lat_col]) if lat_col and pd.notna(row[lat_col]) else None
        lon = float(row[lon_col]) if lon_col and pd.notna(row[lon_col]) else None
        cap = float(row[cap_col]) if cap_col and pd.notna(row[cap_col]) else None
        name = str(row[name_col]) if name_col and pd.notna(row[name_col]) else duid
        meta[duid] = {
            "duid": duid,
            "name": name,
            "lat": lat,
            "lon": lon,
            "capacity_mw": cap
        }
    return meta

# Produce latest output dict per DUID
def latest_output_dict(scada_df, wind_duids=None, agg='last'):
    """
    scada_df: DataFrame with SETTLEMENTDATE (datetime), DUID, SCADAVALUE
    wind_duids: optional list of DUIDs to keep (if None, returns all DUIDs present)
    agg: 'last' or 'max' or 'mean' or custom function - how to select current value per DUID
    """
    df = scada_df.copy()
    df = df.dropna(subset=["DUID", "SCADAVALUE"])
    if wind_duids is not None:
        df = df[df["DUID"].isin(wind_duids)]

    # ensure sorted
    df = df.sort_values("SETTLEMENTDATE")

    if agg == 'last':
        latest = df.groupby("DUID").tail(1).set_index("DUID")["SCADAVALUE"].to_dict()
    elif agg == 'max':
        latest = df.groupby("DUID")["SCADAVALUE"].max().to_dict()
    elif agg == 'mean':
        latest = df.groupby("DUID")["SCADAVALUE"].mean().to_dict()
    else:
        # custom aggregator
        latest = df.groupby("DUID")["SCADAVALUE"].agg(agg).to_dict()

    # Convert numpy types to native Python types
    latest = {k: float(v) if pd.notna(v) else None for k, v in latest.items()}
    return latest

# Combine outputs with predictions and windspeed to create payload for map and /anomaly endpoint
def build_map_payload(models, latest_output, windspeed_dict, windfarm_meta, threshold=0.25):
    """
    models: dict of trained models keyed by DUID
    latest_output: dict {DUID: actual_mw}
    windspeed_dict: dict {DUID: windspeed_m_s}
    windfarm_meta: dict from build_windfarm_meta
    threshold: fraction (e.g., 0.25 = 25%) threshold to flag underperformance

    Returns:
      {
        "predicted": {...},
        "actual": {...},
        "anomalies": {...},  # fractional difference (pred-actual)/pred
        "flagged": {duid: {...meta...}}
        "map_items": [{ name, lat, lon, capacity, actual, predicted, error_pct }]
      }
    """
    predicted = {}
    anomalies = {}
    flagged = {}
    map_items = []

    for duid, meta in windfarm_meta.items():
        # skip if no coords
        lat, lon = meta.get("lat"), meta.get("lon")
        if lat is None or lon is None:
            # can still include but skip map placement
            pass

        actual = latest_output.get(duid, None)
        ws = windspeed_dict.get(duid, None)

        model = models.get(duid)
        if model is None or ws is None:
            pred = None
        else:
            try:
                import numpy as np
                X = np.array(ws).reshape(-1,1) if hasattr(ws, "__iter__") and not isinstance(ws, str) else np.array([[float(ws)]])
                y = model.predict(X)
                # ensure monotone non-negative: we used monotone clipping earlier in model building,
                # but enforce again here: for arrays, cummax & clip; for scalar, clip
                if y.size > 1:
                    y = np.maximum.accumulate(y)
                    y = np.clip(y, 0, None)
                    pred = float(y[-1])
                else:
                    val = float(y[0])
                    pred = max(0.0, val)
            except Exception as e:
                print(f"Prediction failed for {duid}: {e}")
                pred = None

        predicted[duid] = pred
        if pred is None or pred == 0 or actual is None:
            anomalies[duid] = None
        else:
            diff = (pred - actual) / pred
            anomalies[duid] = float(max(0.0, diff))

        is_flagged = anomalies[duid] is not None and anomalies[duid] > threshold
        if is_flagged:
            flagged[duid] = {
                "duid": duid,
                "name": meta.get("name"),
                "lat": lat,
                "lon": lon,
                "capacity_mw": meta.get("capacity_mw"),
                "predicted": predicted[duid],
                "actual": actual,
                "error_frac": anomalies[duid]
            }

        # Build item for map
        map_items.append({
            "duid": duid,
            "name": meta.get("name"),
            "lat": lat,
            "lon": lon,
            "capacity_mw": meta.get("capacity_mw"),
            "windspeed": windspeed_dict.get(duid),
            "actual": actual,
            "predicted": predicted[duid],
            "error_frac": anomalies[duid],
            "flagged": is_flagged
        })

    return {
        "predicted": predicted,
        "actual": latest_output,
        "anomalies": anomalies,
        "flagged": flagged,
        "map_items": map_items
    }

# -----------------------------------------------------------------------------
# Example usage (to adapt in your backend)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) get SCADA dataframe (remote index or local zip dir)
    try:
        scada_df, source = get_latest_scada_df(remote_index_url=None, local_zip_dir="saved_zips")
        print("Loaded SCADA from", source)
    except Exception as e:
        print("Failed to load SCADA:", e)
        raise

    # 2) Build metadata
    meta = build_windfarm_meta(REGISTRATION_CSV)

    # 3) Filter only windfarm DUIDs present in metadata
    wf_duids = set(meta.keys())
    latest = latest_output_dict(scada_df, wind_duids=wf_duids, agg='last')
    print("Latest outputs for", len(latest), "windfarm DUIDs.")

    # 4) Example windspeed dict (you will replace with your Visual Crossing output)
    # e.g., windspeeds = {'RYANCWF1': 8.3, 'ARWF1': 5.6, ...}
    windspeeds = {}

    # 5) Suppose you have already built 'models' dict keyed by DUID
    models = {}  # load your pickled/sklearn pipeline models here

    payload = build_map_payload(models, latest, windspeeds, meta, threshold=0.25)
    print("Map items sample:", payload["map_items"][:3])
