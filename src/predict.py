import pandas as pd
import joblib
import argparse
import yaml
import logging
import os
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

def setup_logger(log_path):
    """ Initialise le logger pour suivre les étapes de prédiction."""
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
def load_config(config_path):
    """ Charge le fichier de configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def preprocess_for_inference(df, encoders, scaler):
    # Supprimer les identifiants inutils
    df= df.drop(columns=['customerID'], errors='ignore')

    # Colonnes numériques à convertir et nettoyer
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

    # Encodage des colonnes catégorielles avec les encoders sauvegardés
    for col, encoder in encoders.items():
        df[col] = df[col].astype(str).fillna('Unknown')
        df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else 'Unknown')
        df[col] = encoder.transform(df[col])

    # Scaler les features
    return scaler.transform(df)

def main(config_path):
    # 1. Chargement de la config et initialisation du logger
    config = load_config(config_path)
    setup_logger(config['logging']['log_file_predict'])
    logging.info("Starting prediction...")

    # 2. Chemins et paramètres depuis la config
    input_path = config['predict']['input_path']
    output_path = config['predict']['output_path']
    model_registry_name= config['predict']['model_registry_name']
    model_version = config['predict']['version_to_use']
    tracking_uri = config['mlflow']['tracking_uri']
    scaler_path = config['model']['scaler_path']
    label_encoder_path = config['model']['label_encoder_path']

    # 3. Chargement des nouvelles données brutes à prédire
    df_raw = pd.read_csv(input_path)
    logging.info(f"Données brutes chargées depuis {input_path}. Shape: {df_raw.shape}")

    # 4. Connexion et chargement du modèle depuis mlflow
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_registry_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    logging.info(f"Modèle chargé depuis {model_uri}.")
    
    # 5. Récupérer le run_id du modèle
    client = MlflowClient()
    model_version_info = client.get_model_version(name=model_registry_name,
                                                  version=model_version)
    print(model_version_info)
    source_run_id = model_version_info.run_id

    # 6. Télécharger les artefacts preprocessing
    preprocessing_dir = "preprocessing_artifacts"
    os.makedirs(preprocessing_dir, exist_ok=True)             
    client.download_artifacts(source_run_id, "preprocessing/scaler.pkl",
                               preprocessing_dir)
    client.download_artifacts(source_run_id, "preprocessing/label_encoders.pkl",
                               preprocessing_dir)
    
    # 7. Charger les objets recupérés depuis mlflow
    scaler = joblib.load(os.path.join(preprocessing_dir, "scaler.pkl"))
    encoders = joblib.load(os.path.join(preprocessing_dir, "label_encoders.pkl"))

    # 8. Prétraitement des nouvelles données
    df_clean = preprocess_for_inference(df_raw, encoders, scaler)
    logging.info("Prétraitement des nouvelles données effectué.")
    
    # 9. Prédiction et sauvegarde des prédictions
    y_pred = model.predict(df_clean)
    df_raw['Churn'] = y_pred

    # 10. Sauvegarde des nouvelles données prédites
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_raw.to_csv(output_path, index=False)
    logging.info(f"Prédictions sauvegardées dans {output_path}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prédiction de churn')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Chemin vers le fichier de configuration')
    args = parser.parse_args()
    main(args.config)

  