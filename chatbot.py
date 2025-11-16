import google.generativeai as genai
import os
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# ⚠️ CONFIGURACIÓN DE API KEY
# Opción 1: Coloca tu API key directamente aquí (más fácil pero menos seguro)
GEMINI_API_KEY = "AIzaSyDP0Y5Bj2Y2fh7yhmnJuB92xXKuqXkoVx4"  # Reemplaza esto con tu API key real

# Opción 2: Usar variable de entorno (más seguro)
# Deja GEMINI_API_KEY = None y configura la variable de entorno en tu sistema


def get_api_key() -> Optional[str]:
    """Obtiene la API key desde el código o variable de entorno."""
    # Primero intenta usar la key del código
    if GEMINI_API_KEY and GEMINI_API_KEY != "TU_API_KEY_AQUI":
        return GEMINI_API_KEY
    # Si no, intenta variable de entorno
    return os.getenv("GEMINI_API_KEY")


def analyze_individual(
    sample_data: pd.Series,
    predicted_prob: float,
    predicted_class: int,
    actual_class: Optional[int] = None,
    shap_values: Optional[np.ndarray] = None,
    top_features: int = 5
) -> str:
    """
    Analiza un individuo usando Gemini AI y proporciona insights.
    
    Args:
        sample_data: Serie de pandas con las características del individuo
        predicted_prob: Probabilidad predicha por el modelo
        predicted_class: Clase predicha (0 o 1)
        actual_class: Clase real del individuo (opcional)
        shap_values: Valores SHAP para las características
        top_features: Número de características principales a destacar
    
    Returns:
        Análisis generado por el chatbot
    """
    
    # Verificar API key
    api_key = get_api_key()
    if not api_key:
        return """⚠️ No se ha configurado GEMINI_API_KEY. 
        
Por favor:
1. Obtén tu API key en: https://aistudio.google.com/app/apikey
2. Colócala en chatbot.py en la variable GEMINI_API_KEY
   O configura la variable de entorno GEMINI_API_KEY"""
    
    try:
        # Configurar Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Preparar información del individuo
        features_text = "\n".join([f"- {name}: {value}" for name, value in sample_data.items()])
        
        # Identificar características más influyentes si hay SHAP values
        influential_features = ""
        if shap_values is not None:
            # Obtener índices de las características más influyentes
            abs_shap = np.abs(shap_values)
            top_indices = np.argsort(abs_shap)[-top_features:][::-1]
            influential_features = "\n\nCaracterísticas más influyentes según SHAP:\n"
            for idx in top_indices:
                feat_name = sample_data.index[idx]
                feat_value = sample_data.iloc[idx]
                shap_val = shap_values[idx]
                direction = "aumenta" if shap_val > 0 else "disminuye"
                influential_features += f"- {feat_name} = {feat_value} ({direction} la probabilidad en {abs(shap_val):.3f})\n"
        
        # Construir prompt
        prompt = f"""Eres un analista experto en interpretación de modelos de machine learning. 

Se te presenta un individuo con las siguientes características:

{features_text}
{influential_features}

El modelo predice:
- Probabilidad de clase positiva: {predicted_prob:.3f}
- Clase predicha: {predicted_class}"""
        
        if actual_class is not None:
            prompt += f"\n- Clase real: {actual_class}"
            if predicted_class != actual_class:
                prompt += " ⚠️ (Predicción INCORRECTA)"
            else:
                prompt += " ✓ (Predicción CORRECTA)"
        
        prompt += """\n\nPor favor, proporciona un análisis conciso (máximo 150 palabras) que:
1. Resuma el perfil del individuo
2. Explique por qué el modelo hizo esta predicción
3. Destaque los factores de riesgo o protección más relevantes
4. Si la predicción es incorrecta, sugiere posibles razones

Usa un tono profesional pero accesible. Responde en español."""

        # Generar respuesta
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"❌ Error al generar análisis: {str(e)}\n\nVerifica que tu API key sea válida y que tengas conexión a internet."


def get_chat_response(user_message: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Función auxiliar para interacción general con el chatbot.
    
    Args:
        user_message: Mensaje del usuario
        context: Contexto adicional (opcional)
    
    Returns:
        Respuesta del chatbot
    """
    api_key = get_api_key()
    if not api_key:
        return "⚠️ No se ha configurado GEMINI_API_KEY."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        response = model.generate_content(user_message)
        return response.text
        
    except Exception as e:
        return f"❌ Error: {str(e)}"
