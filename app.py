import streamlit as st
import pickle
import pandas as pd

# 1. Configuration de la page
st.set_page_config(
    page_title="PredictMind - Santé Mentale des Étudiants",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Injection du CSS personnalisé et de Font Awesome pour des icônes élégantes
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        height: 3.2em;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        opacity: 0.85;
        transform: translateY(-2px);
    }
    .card-low {
        padding: 25px;
        border-radius: 12px;
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid #10b981;
        color: #10b981;
    }
    .card-medium {
        padding: 25px;
        border-radius: 12px;
        background-color: rgba(245, 158, 11, 0.1);
        border: 1px solid #f59e0b;
        color: #f59e0b;
    }
    .card-high {
        padding: 25px;
        border-radius: 12px;
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid #ef4444;
        color: #ef4444;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Chargement du modèle
@st.cache_resource
def charger_modele():
    with open('saved_models/pipeline_sante_mentale.pkl', 'rb') as f:
        modele = pickle.load(f)
    return modele

try:
    model = charger_modele()
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

# 4. En-tête principal
st.markdown("<h1><i class='fa-solid fa-brain' style='color: #6366f1;'></i> PredictMind : Analyse Intelligente</h1>", unsafe_allow_html=True)
st.markdown("### <i class='fa-solid fa-chart-line'></i> Évaluation du niveau de risque et de vulnérabilité de la santé mentale des étudiants", unsafe_allow_html=True)
st.markdown("---")

# 5. Barre latérale (Sidebar) avec tous les paramètres exacts du dataset
st.sidebar.markdown("<h2><i class='fa-solid fa-sliders'></i> Paramètres de l'Étudiant</h2>", unsafe_allow_html=True)
st.sidebar.markdown("Veuillez renseigner les informations ci-dessous :")

def user_input_features():
    col_sb1, col_sb2 = st.sidebar.columns(2)
    with col_sb1:
        age = st.slider("Âge", 17, 30, 20)
        gender = st.selectbox("Genre", ["Male", "Female", "Non-binary"])
        year_of_study = st.selectbox("Année d'étude", [1, 2, 3, 4, 5])
        sleep_hours = st.slider("Sommeil (h/nuit)", 3.0, 10.0, 7.0)
        study_hours = st.slider("Études (h/semaine)", 0.0, 50.0, 15.0)
        gpa = st.slider("GPA", 0.0, 4.0, 3.0)
        social_media = st.slider("Réseaux sociaux (h)", 0.0, 12.0, 3.0)
        screen_time = st.slider("Temps d'écran (h)", 0.0, 16.0, 6.0)

    with col_sb2:
        # Correction : Valeurs exactes du dataset ('Low', 'Moderate', 'High')
        physical_activity = st.selectbox("Activité physique", ["Low", "Moderate", "High"])
        financial_stress = st.slider("Stress financier", 1.0, 10.0, 5.0)
        family_support = st.slider("Soutien familial", 1.0, 10.0, 5.0)
        attendance_rate = st.slider("Taux de présence (%)", 0.0, 100.0, 85.0)
        part_time_job = st.selectbox("Travail partiel", ["Yes", "No"])
        # Correction : Valeur 'Good' au lieu de 'Healthy' pour correspondre au dataset d'origine
        diet_quality = st.selectbox("Alimentation", ["Poor", "Average", "Good"])
        prev_history = st.selectbox("Antécédents médicaux", ["Yes", "No"])

    data = {
        'Age': age,
        'Gender': gender,
        'Year_of_Study': year_of_study,
        'Sleep_Hours': sleep_hours,
        'Study_Hours_Per_Week': study_hours,
        'GPA': gpa,
        'Social_Media_Hours': social_media,
        'Screen_Time_Hours': screen_time,
        'Physical_Activity': physical_activity,
        'Financial_Stress': financial_stress,
        'Family_Support': family_support,
        'Attendance_Rate': attendance_rate,
        'Part_Time_Job': part_time_job,
        'Diet_Quality': diet_quality,
        'Previous_Mental_Health_History': prev_history
    }
    return pd.DataFrame(data, index=[0])

df_input = user_input_features()

# 6. Corps principal
col_main1, col_main2 = st.columns([2, 1])

with col_main1:
    st.markdown("### <i class='fa-solid fa-table-cells'></i> Aperçu des données saisies", unsafe_allow_html=True)
    st.dataframe(df_input, use_container_width=True)

with col_main2:
    st.markdown("### <i class='fa-solid fa-bolt'></i> Action", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("Lancer l'analyse")

# 7. Prédiction et rendu stylisé avec les 3 couleurs distinctes
if predict_button:
    try:
        raw_prediction = model.predict(df_input)[0]
        
        # Mapping sécurisé pour convertir les prédictions numériques ou textuelles
        mapping_labels = {
            0: "Low",
            1: "Medium",
            2: "High"
        }
        
        if isinstance(raw_prediction, (int, float)) or str(raw_prediction).isdigit():
            prediction_label = mapping_labels.get(int(raw_prediction), str(raw_prediction))
        else:
            prediction_label = str(raw_prediction)

        st.markdown("---")
        st.markdown("### <i class='fa-solid fa-square-poll-vertical'></i> Résultat de l'analyse comportementale", unsafe_allow_html=True)
        
        pred_lower = prediction_label.lower()
        
        if "high" in pred_lower:
            st.markdown("""
                <div class="card-high">
                    <h3><i class='fa-solid fa-triangle-exclamation'></i> Niveau de risque : ÉLEVÉ (HIGH)</h3>
                    <p>L'étudiant présente des indicateurs de vulnérabilité importants nécessitant une attention ou un accompagnement prioritaire.</p>
                </div>
            """, unsafe_allow_html=True)
        elif "med" in pred_lower:
            st.markdown("""
                <div class="card-medium">
                    <h3><i class='fa-solid fa-circle-exclamation'></i> Niveau de risque : MODÉRÉ (MEDIUM)</h3>
                    <p>L'étudiant montre certains signaux de stress ou de déséquilibre à surveiller de près.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="card-low">
                    <h3><i class='fa-solid fa-circle-check'></i> Niveau de risque : FAIBLE (LOW)</h3>
                    <p>L'étudiant présente un profil stable sans alerte majeure détectée.</p>
                </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Une erreur est intervenue lors du calcul de la prédiction : {e}")