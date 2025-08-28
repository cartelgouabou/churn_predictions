import os
import glob
import pandas as pd
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/outputs")
PRED_FILE_PATTERN = os.getenv("PRED_FILE_PATTERN", "predictions_*.csv")

app = FastAPI(title="Churn API (read-only from outputs)", version="0.1.0")

def _latest_predictions_path() -> str:
    paths = sorted(glob.glob(os.path.join(OUTPUT_DIR, PRED_FILE_PATTERN)))
    if not paths:
        raise FileNotFoundError(
            f"Aucun fichier de prédiction trouvé dans {OUTPUT_DIR} avec le pattern {PRED_FILE_PATTERN}."
        )
    return paths[-1]

@app.get("/health")
def health():
    exists = bool(glob.glob(os.path.join(OUTPUT_DIR, PRED_FILE_PATTERN)))
    return {"status": "ok" if exists else "no_predictions_yet"}

@app.get("/at_risk")
def at_risk(min_proba: float = 0.5) -> List[Dict[str, Any]]:
    """
    Renvoie les clients à risque (churn_pred == 1 et churn_proba >= min_proba)
    Champs: customerID, gender, churn_proba, churn_pred
    """
    try:
        path = _latest_predictions_path()
        df = pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lecture prédictions impossible : {e}")

    expected = {"customerID", "gender", "churn_proba", "churn_pred"}
    missing = expected - set(df.columns)
    if missing:
        raise HTTPException(status_code=500, detail=f"Colonnes manquantes dans {path}: {sorted(missing)}")

    out = (
        df[(df["churn_pred"] == 1) & (df["churn_proba"] >= min_proba)]
        [["customerID", "gender", "churn_proba", "churn_pred"]]
        .sort_values("churn_proba", ascending=False)
        .copy()
    )
    return out.to_dict(orient="records")
