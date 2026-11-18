import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "student-mental-health.csv"



def executer_preprocessing():
    print("--- Debut du pretraitement des donnees ---")

    df = pd.read_csv(DATA_PATH)
    
    
    cols_a_ignorer = ['Student_ID', 'Mental_Health_Risk']
    colonnes_features = [col for col in df.columns if col not in cols_a_ignorer]
    
    X = df[colonnes_features]
    y = df['Mental_Health_Risk']

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    numeric_cols = X.select_dtypes(include="number").columns.to_list()
    nominal_cols = ['Gender', 'Part_Time_Job', 'Previous_Mental_Health_History']
    ordinal_cols = ['Diet_Quality', 'Physical_Activity']

    categories_ordinales = [
        ["Poor", "Average", "Good"],
        ["Low", "Moderate", "High"]   
    ]

    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),      
        ('scaler', StandardScaler())                    
    ])

    nominal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(handle_unknown="ignore", sparse_output=False)) 
    ])

    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('ordinal', OrdinalEncoder(categories=categories_ordinales)) 
    ])

    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('nom', nominal_transformer, nominal_cols),
            ('ord', ordinal_transformer, ordinal_cols)
        ]
    )


    os.makedirs('saved_models', exist_ok=True)
    chemin_sortie = 'saved_models/donnees_et_preprocesseur.pkl'
    
    with open(chemin_sortie, 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test, preprocessor), f)
        
    print(f"Pretraitement termine et les Donnees sauvegardees dans {chemin_sortie}")

if __name__ == "__main__":
    executer_preprocessing()