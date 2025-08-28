import os
import sys
import pandas as pd
import numpy as np
import joblib
import argparse
import logging
from datetime import datetime

def setup_logger(log_dir= "/app/logs"):
    """ Initialise le logger pour suivre les étapes de prédiction."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "predict.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    
def load_artifacts(actifacts_dir):
    model_path = os.path.join(actifacts_dir, "xgb_model.pkl")
    scaler_path = os.path.join(actifacts_dir, "scaler.pkl")
    encoders_path = os.path.join(actifacts_dir, "label_encoders.pkl")
    if not all(os.path.exists(path) for path in [model_path, scaler_path, encoders_path]):
        raise FileNotFoundError(
            f"Artefacts manquants dans {actifacts_dir}. "
            "Attendu: xgb_model.pkl, scaler.pkl, label_encoders.pkl"
        )
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoders = joblib.load(encoders_path)
    return model, scaler, encoders 
    
def preprocess_for_inference(df, encoders, scaler):
    # Supprimer les identifiants inutils
    df= df.drop(columns=['customerID'], errors='ignore')

    # Colonnes numériques à convertir et nettoyer
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numerical_cols:
        if col not in df.columns:
            raise ValueError(f"Colonne numérique attendue manquante : {col}.")
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

    # Encodage des colonnes catégorielles avec les encoders sauvegardés
    for col, encoder in encoders.items():
        if col not in df.columns:
            raise ValueError(f"Colonne catégorie attendue manquante : {col}.")
        df[col] = df[col].astype(str).fillna('Unknown')
        df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else 'Unknown')
        df[col] = encoder.transform(df[col])

    if not hasattr(scaler, "feature_names_in_"):
        raise RuntimeError("" \
        "Le scaler ne contient pas 'feature_names_in_'. " 
        "Assurez-vous qu'il a été fit sur un Dataframe avec les noms de colonnes")

    required = list(scaler.feature_names_in_)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Les colonnes suivantes sont manquantes : {missing}.")
    
    # Réordonnéer strictiement selon le scaler
    df = df[required]
    # Scaler les features
    return scaler.transform(df)

def main():
    setup_logger()
    parser = argparse.ArgumentParser(description='Prédiction de churn')
    parser.add_argument('--input_csv', default = "/app/data/raw_data_churn_test.csv", help='Chemin vers le fichier CSV contenant les données de test')
    parser.add_argument("--artifacts_dir", default="/app/models", help="Chemin vers les artefacts du modèle")
    parser.add_argument("--output_dir", default="/app/outputs", help="Chemin vers le dossier de sortie")
    args = parser.parse_args()

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        model, scaler, encoders = load_artifacts(args.artifacts_dir)
        df_to_predict = pd.read_csv(args.input_csv)
        X_scaled = preprocess_for_inference(df_to_predict, encoders, scaler)
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        preds_label = (y_pred_proba>=0.5).astype(int)
        # Horodaté les sorties
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.output_dir, f"predictions_{ts}.csv")
        out_df = df_to_predict.copy()
        out_df["churn_proba"] = y_pred_proba    
        out_df["churn_pred"] = preds_label
        out_df.to_csv(out_path, index=False)
        logging.info(f"Prédictions enregistrées à {out_path}")
    except Exception as e:
        logging.error(f"Une erreur s'est produite lors de la prédiction : {e}")
        sys.exit(1)



if __name__ == '__main__':
    main()

  