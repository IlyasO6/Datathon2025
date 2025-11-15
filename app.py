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

if pagina == 'Resumen Global':
    st.header('Insights Globales del Modelo')
    st.write("Esta vista muestra qué factores impulsan las predicciones en general.")

    # A. Rendimiento del Modelo (25% de la nota) [cite: 65]
    st.subheader('Rendimiento del Modelo')
    # Debes tener tu F1 score guardado, o calcularlo aquí.
    # El mínimo es 0,7 [cite: 43]
    st.metric(label="F1 Score (Test)", value=0.78) # Reemplaza 0.78 con tu valor real
    # Aquí podrías mostrar una matriz de confusión (con st.pyplot)

    # B. SHAP Summary Plot (Global Explainability)
    st.subheader('Importancia General de las Features')
    
    # Usamos st.pyplot() para "atrapar" el gráfico de matplotlib
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig)

    st.subheader('Impacto Detallado de Features')
    fig2, ax2 = plt.subplots()
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig2)