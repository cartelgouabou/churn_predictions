Feature Prédiction du Churn Client – Pipeline MLOps Industrialisable

## Présentation

Ce projet a pour objectif de développer un pipeline complet de **machine learning** pour la **prédiction du churn client** à partir de données structurées. Il a été conçu dans un contexte d’entreprise pour répondre aux exigences de robustesse, traçabilité, maintenabilité et industrialisation.

Le pipeline s’appuie sur deux modèles supervisés :
- Un **modèle de référence** : régression logistique
- Un **modèle de production** : XGBoost, optimisé pour la performance

Le projet suit rigoureusement les **bonnes pratiques MLOps** et est prêt pour une intégration avec des outils comme **MLflow**, **DVC** ou des plateformes CI/CD.

---

## Structure du Répertoire

```
churn_prediction_project/
│
├── configs/                                     # Fichiers de configuration YAML
│   ├── config.yaml                              # Données brutes
├── data/                 
│   ├── Telco-Customer-Churn.csv                # Données brutes
│   └── clean_data.csv                          # Données nettoyées
├── models/                                     # Modèles entraînés (fichiers .pkl)
├── logs/                                       # Logs d’exécution
├── outputs/                                    # Prédictions générées
├── notebooks/                                  # Notebooks exploratoires
├── src/                                        # Scripts Python modulaires
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── utils.py
├── tests/                                      # Tests unitaires
│   ├── test_data_coherence.py
├── run_all.sh
├── requirements.txt
└── README.md
```

---

## Mise en place

### 1. Cloner le projet

```bash
git clone https://github.com/cartelgouabou/churn_predictions.git
cd churn_predictions
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv-churn
```

### 3. Activer l’environnement virtuel
- Sur Linux / macOS: 

```bash
source venv-churn/bin/activate
```
- Sur Windows :
```bash
venv-churn\Scripts\activate
```

### 4. Installer les dépendances Python

```bash
pip install -r requirements.txt
```

> Version Python recommandée : 3.8 ou supérieure

---

## Exécution du pipeline

### Exécution complète du pipeline

```bash
./run_all.sh
```

### Génération de prédictions sur de nouvelles données

Placer le fichier d'entrée dans `data/new_customers.csv`, puis exécuter :
```bash
python3 src/predict.py --config config/config.yaml
```
Les résultats seront enregistrés dans `outputs/predictions.csv`.

---

## Tests unitaires

Lancer les tests avec :

```bash
pytest tests/
```

**Tests disponibles :**
- `test_data_coherence.py` : vérifie l’intégrité structurelle des données traitées

---

## Configuration

Tous les paramètres de traitement, chemins de fichiers et hyperparamètres sont centralisés dans :
```yaml
configs/config.yaml
```
Modifier ce fichier permet d’adapter le pipeline sans toucher aux scripts Python.

---

## Bonnes Pratiques MLOps Respectées

- **Scripts découplés** : un script = une tâche
- **Paramétrage centralisé** via YAML
- **Exécution CLI** via `argparse`
- **Logging** professionnel (pas de `print()`)
- **Tests unitaires** intégrés
- **Prêt pour DVC / MLflow / CI-CD**

---

## Prochaines étapes 

- Intégration de **MLflow** pour le suivi des expériences
- Versionnage des données et des modèles via **DVC**
- Création d’une API REST avec **FastAPI** pour le scoring en temps réel
- Déploiement CI/CD avec **GitHub Actions** ou **GitLab CI**
- Détection de dérive et automatisation de la réentraînement

---

## Contact

Pour toute question relative au projet, technique ou académique :
- 📧 cartelgouabou@gmail.com
- 🌐 cartelgouabou.github.io
- 🐙 GitHub : github.com/cartelgouabou

---

© 2025 Arthur Cartel Foahom Gouabou — Projet réalisé dans le cadre de la formation "MLOps : Le guide complet pour réussir le déploiement d’un modèle IA". 
