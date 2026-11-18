import os
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def entrainer_modeles():
    print("--- 1. Etape d'entrainement et d'optimisation ---")
    
   
    with open('saved_models/donnees_et_preprocesseur.pkl', 'rb') as f:
        X_train, X_test, y_train, y_test, preprocessor = pickle.load(f)

    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    
    modeles_a_optimiser = {
        "Logistic Regression": {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {'classifier__C': [0.01, 0.1, 1, 10, 100]}
        },
        "Random Forest": {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20, 30, None],
                'classifier__min_samples_split': [2, 5, 10]
            }
        },
        "XGBoost": {
            'model': XGBClassifier(random_state=42, eval_metric='mlogloss'),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.01, 0.05, 0.1],
                'classifier__max_depth': [3, 5, 7]
            }
        }
    }

    meilleurs_modeles = {}
    resultats_comparaison = []

    
    for nom, config in modeles_a_optimiser.items():
        print(f"Optimisation en cours : {nom}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', config['model'])
        ])
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=config['params'],
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train_encoded)
        
        meilleurs_modeles[nom] = grid_search.best_estimator_
        resultats_comparaison.append({
            "Modele": nom,
            "Meilleurs Parametres": grid_search.best_params_,
            "Score CV (Accuracy)": grid_search.best_score_
        })

    
    df_resultats = pd.DataFrame(resultats_comparaison)
    print("\n--- COMPARISON DES MODÈLES (Validation Croisée) ---")
    print(df_resultats.to_string(index=False))

    
    meilleur_nom_modele = df_resultats.loc[df_resultats["Score CV (Accuracy)"].idxmax()]["Modele"]
    modele_optimal = meilleurs_modeles[meilleur_nom_modele]
    print(f"\n Modele optimal selectionne : {meilleur_nom_modele}")

    
    os.makedirs('saved_models', exist_ok=True)
    chemin_modele_final = 'saved_models/pipeline_sante_mentale.pkl'
    with open(chemin_modele_final, 'wb') as f:
        pickle.dump(modele_optimal, f)
        
    print(f"Pipeline sauvegarde dans {chemin_modele_final}")
    return meilleur_nom_modele


def evaluer_modele():
    print("\n--- 2. etape d'evaluation sur le test ---")
    
    
    with open('saved_models/donnees_et_preprocesseur.pkl', 'rb') as f:
        X_train, X_test, y_train, y_test, preprocessor = pickle.load(f)

    with open('saved_models/pipeline_sante_mentale.pkl', 'rb') as f:
        modele_optimal = pickle.load(f)

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)              
    y_test_encoded = label_encoder.transform(y_test) 

    y_pred_optimal = modele_optimal.predict(X_test)
    acc_test = accuracy_score(y_test_encoded, y_pred_optimal)
    
    print(f"Accuracy sur le Test : {acc_test:.4f}")
    print("\nRapport de classification détaillé :")
    print(classification_report(y_test_encoded, y_pred_optimal, target_names=label_encoder.classes_))

if __name__ == "__main__":
   
    entrainer_modeles()
    evaluer_modele()