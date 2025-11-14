import json
from pathlib import Path

COUNTER_FILE = Path("app/metadata/anomaly_total.json")

def update_anomaly_total(df):
    # Load existing total
    if COUNTER_FILE.exists():
        total = json.loads(COUNTER_FILE.read_text())["total"]
    else:
        total = 0.0

    # Calculate this refresh's anomaly MW
    for _, row in df.iterrows():
        expected = row["Expected (MW)"]
        actual = row["Actual (MW)"]
        if expected and actual and expected > actual:
            deficit = expected - actual
            # only count if error > 20%
            if row["Error (%)"] and row["Error (%)"] > 20:
                total += deficit

    # Save updated total
    COUNTER_FILE.write_text(json.dumps({"total": total}))

    return total