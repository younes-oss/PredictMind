
from datetime import datetime, timedelta
import sys
import os

# Imports Airflow 3
from airflow.sdk import DAG, task


# Ajouter le dossier src au chemin de recherche Python
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../src')
    )
)

# Importer les fonctions de preprocessing et d'entraînement
from preprocessing_script import executer_preprocessing
from train_model_script import entrainer_modeles, evaluer_modele


# Configuration des tâches
default_args = {
    'owner': 'predictmind_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# Définition du DAG
with DAG(
    dag_id='predictmind_pipeline_sante_mentale',
    default_args=default_args,
    description='Pipeline automatisé de prédiction du risque de santé mentale',
    schedule='@weekly',
    catchup=False,
) as dag:

    # Tâche 1 : Prétraitement des données
    @task(task_id='pre_traitement_donnees')
    def task_preprocessing():
        return executer_preprocessing()

    # Tâche 2 : Entraînement des modèles
    @task(task_id='entrainement_modeles_gridsearch')
    def task_entrainement():
        return entrainer_modeles()

    # Tâche 3 : Évaluation du meilleur modèle
    @task(task_id='evaluation_modele_optimal')
    def task_evaluation():
        return evaluer_modele()

    # Création des tâches
    t1 = task_preprocessing()
    t2 = task_entrainement()
    t3 = task_evaluation()

    # Définition de l'ordre d'exécution
    t1 >> t2 >> t3
