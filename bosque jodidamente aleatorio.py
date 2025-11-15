import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import shap  # Asegúrate de haber hecho 'pip install shap'
import warnings

# --- 0. Configuración ---
warnings.filterwarnings('ignore') # Ocultar advertencias

# --- 1. Cargar tu dataset ---
try:
    datos = pd.read_csv("dataset.csv")
    print("Dataset 'dataset.csv' cargado con éxito.")
except FileNotFoundError:
    print("Error: No se pudo encontrar el archivo 'dataset.csv'.")
    exit()

# --- 2. Definir X (features) e y (target) ---
nombre_columna_objetivo = 'target_variable' 
lista_de_predictores = [
    'product_A_sold_in_the_past', 'product_B_sold_in_the_past',
    'product_A_recommended', 'product_A', 'product_C', 'product_D',
    'cust_hitrate', 'cust_interactions', 'cust_contracts', 'opp_month',
    'opp_old', 'competitor_Z', 'competitor_X', 'competitor_Y', 'cust_in_iberia'
]

X = datos[lista_de_predictores]
y = datos[nombre_columna_objetivo]

print(f"\nPrediciendo '{nombre_columna_objetivo}' usando {len(lista_de_predictores)} predictores.")

# --- 3. Dividir los datos ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=30, stratify=y
)
print(f"Datos divididos: {len(X_train)} filas para entrenamiento, {len(X_test)} filas para prueba.")

# --- 4. Crear y Entrenar el Modelo Random Forest ---
print("\nEntrenando Random Forest...")
modelo_rf = RandomForestClassifier(
    n_estimators=500,
    random_state=30,
    n_jobs=-1,
    class_weight="balanced"
)
modelo_rf.fit(X_train, y_train)
print("¡Modelo entrenado!")

# --- 5. Ver la Importancia de las Variables (Método Rápido) ---
print("\n--- Importancia de Variables (Gini) ---")
importancias = modelo_rf.feature_importances_
df_importancia = pd.DataFrame({'Variable': lista_de_predictores, 'Importancia': importancias})
df_importancia = df_importancia.sort_values(by='Importancia', ascending=False)
print(df_importancia.to_string(index=False))

# --- 6. Calcular SHAP (¡LA VERSIÓN RÁPIDA!) ---
# print("\nCalculando valores SHAP (sobre una muestra de 2000 filas)...")
# explainer = shap.TreeExplainer(modelo_rf)
# X_test_sample = X_test.sample(n=20, random_state=30, replace=False) 
# shap_values_sample = explainer.shap_values(X_test_sample)
# print("Valores SHAP calculados. Generando gráfico resumen...")

# # For binary classification, use the SHAP values for the positive class
# # Check if shap_values is a list (multi-output) or array (single output)
# if isinstance(shap_values_sample, list):
#     shap.summary_plot(shap_values_sample[1], X_test_sample)
# else:
#     shap.summary_plot(shap_values_sample, X_test_sample)

# print("Gráfico SHAP generado.")

# --- 7. Evaluar el Modelo (CON UMBRAL FIJO 0.38) ---
print("\nCalculando F1-Score con el umbral 0.38...")

# Obtener las probabilidades de la clase positiva (Won = 1)
probabilidades = modelo_rf.predict_proba(X_test)[:, 1]

# Definir tu umbral fijo
umbral_fijo = 0.38

# Aplicar el umbral para obtener las predicciones
predicciones_con_umbral_fijo = (probabilidades >= umbral_fijo).astype(int)

# Calcular el F1-Score con esas predicciones
f1_final = f1_score(y_test, predicciones_con_umbral_fijo)

print("\n--- Resultados Finales (Random Forest) ---")
print(f"F1 Score usando el umbral de {umbral_fijo}: {f1_final:.4f}")