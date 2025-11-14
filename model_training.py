import pandas as pd
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

REG_CSV = Path("data/Merged.csv")
MODEL_DIR = Path("app/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def build_models_monotone_nonneg(combined_dfs, degree=3):
    """
    Build polynomial regression models for each windfarm with:
    - monotone increasing output from 0 m/s
    - all predictions >= 0 MW
    """
    models = {}
    for wf_name, df in combined_dfs.items():
        sub = df.dropna(subset=['windspeed', 'output MW'])
        X = sub['windspeed'].values.reshape(-1, 1)
        y = sub['output MW'].values

        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X, y)
        models[wf_name] = model

    return models

def save_models(models):
    for wf_name, model in models.items():
        with open(MODEL_DIR / f"{wf_name}.pkl", "wb") as f:
            pickle.dump(model, f)

def main():
    df = pd.read_csv(REG_CSV)
    # Assume one CSV per windfarm; or split by DUID
    combined_dfs = {row['duid']: df[df['duid']==row['duid']] for _, row in df.iterrows()}
    models = build_models_monotone_nonneg(combined_dfs)
    save_models(models)
    print("✅ Models built and saved.")

if __name__ == "__main__":
    main()
