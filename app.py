import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt # Importante para los gráficos de SHAP




@st.cache_resource
def load_model_and_explainer():
    model = joblib.load('utils/modelo.joblib')
    explainer = joblib.load('utils/shap_explainer.joblib')
    return model, explainer

# st.cache_data es para datos, como dataframes o arrays
@st.cache_data
def load_data_and_shap():
    shap_values = joblib.load('utils/shap_values.joblib')
    return shap_values

# Carga todo al inicio
model, explainer = load_model_and_explainer()
shap_values = load_data_and_shap()

# --- Fin de la Carga ---

st.title('Dashboard de Explainability (GTM)')


st.sidebar.title('Navegación')
pagina = st.sidebar.radio('Selecciona una vista:', ['Resumen Global', 'Análisis Local de Oportunidad'])
# (Aquí irá la lógica del dashboard, ver Paso 3)