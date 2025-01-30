# Projet de Prédiction des Maladies Rénales

Ce projet vise à prédire la nécessité de dialyse chez les patients atteints de maladies rénales en utilisant des techniques d'apprentissage automatique. Les données utilisées pour l'entraînement et la prédiction proviennent du fichier `data/kidney_disease.csv`.

## Structure du Projet

- **data/** : Contient les données brutes.
  - `kidney_disease.csv` : Données sur les maladies rénales.
  
- **notebooks/** : Contient des notebooks Jupyter pour l'analyse des données.
  - `data_analysis.ipynb` : Analyse exploratoire des données avec visualisations et statistiques descriptives.
  
- **src/** : Contient le code source pour le prétraitement des données, l'entraînement du modèle, l'évaluation et la prédiction.
  - `data_preprocessing.py` : Fonctions pour le nettoyage et la normalisation des données.
  - `model_training.py` : Code pour entraîner le modèle de prédiction.
  - `model_evaluation.py` : Évaluation des performances du modèle.
  - `prediction.py` : Fonctions pour effectuer des prédictions avec le modèle entraîné.
  
- **models/** : Contient le modèle entraîné.
  - `trained_model.pkl` : Modèle sauvegardé au format pickle.
  
- **requirements.txt** : Liste des dépendances nécessaires pour exécuter le projet.

## Instructions

1. **Configuration de l'environnement** :
   - Clonez le dépôt et naviguez dans le répertoire du projet.
   - Installez les dépendances en exécutant :
     ```
     pip install -r requirements.txt
     ```

2. **Exécution des scripts** :
   - Utilisez `data_preprocessing.py` pour préparer les données.
   - Exécutez `model_training.py` pour entraîner le modèle.
   - Évaluez le modèle avec `model_evaluation.py`.
   - Utilisez `prediction.py` pour faire des prédictions sur de nouvelles données.

3. **Utilisation du modèle** :
   - Chargez le modèle à partir de `models/trained_model.pkl` pour effectuer des prédictions sur la nécessité de dialyse.

## Auteurs

- [Votre Nom] - [Votre Email]

## License

Ce projet est sous licence MIT. Veuillez consulter le fichier LICENSE pour plus de détails.