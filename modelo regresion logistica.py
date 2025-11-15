import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # ¡Importante para buenas prácticas!

# --- 1. Cargar tu dataset ---
try:
    datos = pd.read_csv("dataset.csv")
    print("Dataset 'dataset.csv' cargado con éxito.")
except FileNotFoundError:
    print("Error: No se pudo encontrar el archivo 'dataset.csv'.")
    exit()

# --- 2. Definir X (features) e y (target) ---

# Variable objetivo (lo que quieres predecir)
nombre_columna_objetivo = 'target_variable' 

# Lista de todos los predictores (todas las columnas MENOS 'id' y 'target_variable')
lista_de_predictores = [
    'product_A_sold_in_the_past', 'product_B_sold_in_the_past',
    'product_A_recommended', 'product_A', 'product_C', 'product_D',
    'cust_hitrate', 'cust_interactions', 'cust_contracts', 'opp_month',
    'opp_old', 'competitor_Z', 'competitor_X', 'competitor_Y', 'cust_in_iberia'
]

X = datos[lista_de_predictores]
y = datos[nombre_columna_objetivo]

print(f"\nPrediciendo '{nombre_columna_objetivo}' usando {len(lista_de_predictores)} predictores.")

# --- 3. Dividir los datos (¡Buena práctica!) ---
# Usamos 70% para entrenar y 30% para probar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Datos divididos: {len(X_train)} filas para entrenamiento, {len(X_test)} filas para prueba.")

# Importa el escalador
from sklearn.preprocessing import StandardScaler

# --- 3.5. ¡NUEVO PASO! ESCALAR LOS DATOS ---
print("Escalando las variables...")

# 1. Crea el objeto escalador
scaler = StandardScaler()

# 2. AJÚSTALO (fit) SÓLO con los datos de entrenamiento
scaler.fit(X_train)

# 3. TRANSFORMA ambos (entrenamiento y prueba)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. Crear y Entrenar el Modelo (Con datos escalados) ---
print("Entrenando el modelo de Regresión Logística...")

modelo = LogisticRegression(max_iter=1000) 
# ¡Entrena con X_train_scaled!
modelo.fit(X_train_scaled, y_train)

print("¡Modelo entrenado!")

# --- 6. Evaluar el Modelo (Básico) ---
print("Evaluando el modelo...")

from sklearn.metrics import f1_score  # <-- Agrega esta importación

# ¡Obtén probabilidades usando X_test_scaled!
probabilidades = modelo.predict_proba(X_test_scaled)[:, 1]

# (El resto de tu código de umbral y F1 funciona igual)
umbral_personalizado = 0.4 
predicciones_personalizadas = (probabilidades >= umbral_personalizado).astype(int)
f1_personalizado = f1_score(y_test, predicciones_personalizadas)

print(f"\n--- Resultados con umbral personalizado = {umbral_personalizado} ---")
print(f"F1 Score: {f1_personalizado:.4f}")