import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import streamlit as st


# ----------------------------
# Configuración de la página
# ----------------------------
st.set_page_config(page_title="Dashboard de Modelo", layout="wide")


# ----------------------------
# Estilos (Tema Verde Ecológico)
# ----------------------------
st.markdown(
    """
    <style>
    :root {
        --bg-main: #159921;
        --bg-card: #B9C9B9;
        --text-main: #FFFFFF;
        --primary: #2E7D32;
        --primary-dark: #1B5E20;
        --primary-soft: #A5D6A7;
        --accent: #FBC02D;
    }

    /* Fondo general de la app */
    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg-main) !important;
        color: var(--text-main) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid rgba(0,0,0,0.06);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--primary-dark) !important;
    }

    /* Títulos */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-main) !important;
    }

    /* Header principal tipo card */
    .bg-card {
        background: linear-gradient(90deg, #1B5E20, #33691E);
        padding: 1.75rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 18px rgba(0,0,0,0.15);
        border-radius: 12px;
        border: none;
    }
    .bg-card h1 {
        margin: 0;
        color: #B9C9B9;
        font-size: 2.6rem;
        font-weight: 700;
    }
    .bg-card p {
        margin-top: 0.4rem;
        margin-bottom: 0;
        color: #E8F5E9;
        font-size: 1rem;
    }

    /* Tarjetas / contenedores */
    .card {
        background: var(--bg-card);
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(15, 32, 24, 0.08);
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(0,0,0,0.03);
    }

    /* Botones */
    div.stButton > button,
    .stDownloadButton > button,
    .st-emotion-cache-1vt4y43 {
        background-color: var(--primary) !important;
        color: #FFFFFF !important;
        border-radius: 999px !important;
        border: none !important;
        padding: 0.45rem 1.2rem;
        font-weight: 600;
    }
    div.stButton > button:hover,
    .stDownloadButton > button:hover {
        background-color: var(--primary-dark) !important;
    }

    /* Select / inputs */
    .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 10px !important;
        border-color: rgba(0,0,0,0.10) !important;
        background-color: #FFFFFF !important;
    }

    /* Métrica destacada */
    [data-testid="stMetricValue"] {
        color: var(--primary) !important;
        font-weight: 700 !important;
    }

    /* Pequeño acento en enlaces */
    a {
        color: var(--primary) !important;
    }
    a:hover {
        color: var(--accent) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Utilidades de carga y helpers
# ----------------------------
ROOT = Path(".")


def safe_load_pickle(path: Path):
	with open(path, "rb") as f:
		try:
			return pickle.load(f)
		except Exception:
			# fallback a joblib si fuese necesario sin agregar dependencia dura
			try:
				import joblib  # type: ignore
				f.seek(0)
				return joblib.load(f)
			except Exception as e:  # pragma: no cover
				raise e


def ensure_metrics_csv(classif_report_json: Path, metrics_csv: Path) -> pd.DataFrame:
	if not classif_report_json.exists():
		st.error(f"No se encontró '{classif_report_json}'. Por favor, colócalo en el directorio de trabajo.")
		return pd.DataFrame()

	with open(classif_report_json, "r", encoding="utf-8") as f:
		report = json.load(f)

	df_metrics = pd.DataFrame(report).T
	# Normalizar nombres de columnas si fuese necesario
	df_metrics.columns = [c.replace(" ", "_") for c in df_metrics.columns]
	# Guardar CSV
	df_metrics.to_csv(metrics_csv, index=True)
	return df_metrics


def emoji_for_shap(val: float) -> str:
	if val > 0.5:
		return "🤑"
	if val < -0.5:
		return "💀"
	if 0 < val <= 0.5:
		return "😊"
	if -0.5 <= val < 0:
		return "😟"
	return "😶"


def pick_sample_id_options(X: pd.DataFrame) -> pd.Series:
	if "id" in X.columns:
		return X["id"]
	# fallback: usar índice como ID
	return pd.Series(X.index, index=X.index, name="id")


# -----------------------------------------
# Cargar datos/artefactos requeridos (perezoso)
# -----------------------------------------
MODEL_PKL = ROOT / "model.pkl"
EXPLAINER_PKL = ROOT / "explainer.pkl"
SHAP_VALUES_PKL = ROOT / "shap_values.pkl"
X_TEST_CSV = ROOT / "X_test.csv"
CLASSIF_JSON = ROOT / "classification_report.json"
METRICS_CSV = ROOT / "metrics.csv"
ROC_JSON = ROOT / "roc_curve_data.json"
ARTIFACTS_XGB_DIR = ROOT / "artifacts_xgb"


@st.cache_data(show_spinner=False)
def load_x_test() -> pd.DataFrame:
	return pd.read_csv(X_TEST_CSV)


@st.cache_data(show_spinner=False)
def load_metrics_df() -> pd.DataFrame:
	if METRICS_CSV.exists():
		return pd.read_csv(METRICS_CSV, index_col=0)
	return ensure_metrics_csv(CLASSIF_JSON, METRICS_CSV)


@st.cache_resource(show_spinner=False)
def load_explainer():
	return safe_load_pickle(EXPLAINER_PKL)


@st.cache_resource(show_spinner=False)
def load_model():
	return safe_load_pickle(MODEL_PKL)


@st.cache_resource(show_spinner=False)
def load_shap_values():
	vals = safe_load_pickle(SHAP_VALUES_PKL)
	if isinstance(vals, pd.DataFrame):
		vals = vals.values
	return np.asarray(vals)


# ----------------------------
# Título principal
# ----------------------------


# ----------------------------
# Tabs principales
# ----------------------------
# ---- MENÚ LATERAL ----
with st.sidebar:
	st.title("Dashboard")
	opcion = st.radio(
		"Secciones",
		(
			"Resultados (User-Friendly)",
			"Análisis Técnico (SHAP)",
			"Validez del Modelo (Métricas)"
		)
	)
# ----------------------------
# Tab 1: Resultados (User-Friendly)
# ----------------------------
if opcion == "Resultados (User-Friendly)":
	st.subheader("Gráficos SHAP")

	missing = []
	for p in [EXPLAINER_PKL, SHAP_VALUES_PKL, X_TEST_CSV]:
		if not p.exists():
			missing.append(str(p))
	if missing:
		st.error("Faltan archivos necesarios: " + ", ".join(missing))
		st.stop()

	_ = load_explainer()  # cargado aunque no se use estrictamente para summary_plot
	X_test = load_x_test()
	shap_values = load_shap_values()

	col1, col2 = st.columns(2)

	with col1:
		# Beeswarm
		st.markdown("**Beeswarm**")
		fig1 = plt.figure(figsize=(10, 4))
		shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
		st.pyplot(fig1, use_container_width=True)
		plt.close(fig1)

	with col2:
		# Bar
		st.markdown("**Importancia (Bar Plot)**")
		fig2 = plt.figure(figsize=(10, 4))
		shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
		st.pyplot(fig2, use_container_width=True)
		plt.close(fig2)


# ----------------------------
# Tab 2: Análisis Técnico (SHAP)
# ----------------------------
elif opcion == "Análisis Técnico (SHAP)":
	st.subheader("Análisis Técnico (SHAP)")

	artifacts_dir = ARTIFACTS_XGB_DIR
	if not artifacts_dir.exists():
		st.error(f"No se encontró el directorio de artefactos: '{artifacts_dir}'.")
	else:
		# Mostrar Feature Importance primero
		fi_path = artifacts_dir / "feature_importance_xgb.png"
		if fi_path.exists():
			st.image(str(fi_path), caption="Feature Importance (XGB)", use_container_width=True)
		else:
			st.info("No se encontró 'feature_importance_xgb.png' en artifacts_xgb.")

		# Listado de dependence plots
		dep_pngs = sorted([p for p in artifacts_dir.glob("*.png") if "dependence" in p.stem.lower()])
		if not dep_pngs:
			st.info("No se encontraron gráficos de dependencia SHAP en artifacts_xgb.")
		else:
			options = [p.name for p in dep_pngs]
			selected = st.selectbox("Selecciona un SHAP dependence plot", options=options, index=0)
			st.image(str(artifacts_dir / selected), caption=selected, use_container_width=True)


# ----------------------------
# Tab 3: Validez del Modelo (Métricas)
# ----------------------------
elif opcion == "Validez del Modelo (Métricas)":

	#  Edited
	# Layout en dos columnas: más espacio para la ROC
	col_metric, col_roc = st.columns([1, 2.2])

	with col_metric:
		# Título estilizado (blanco y un poco más grande)
		st.markdown(
			"<h4 style='color:#FFFFFF; margin:0 0 0.5rem 0;'>F1-Score </h4>",
			unsafe_allow_html=True
		)
		df_metrics = load_metrics_df()
		if df_metrics.empty:
			st.error("No se pudieron cargar las métricas. Verifica 'classification_report.json'.")
		else:
			f1_col = "f1-score" if "f1-score" in df_metrics.columns else ("f1_score" if "f1_score" in df_metrics.columns else None)
			f1_weighted = None
			if f1_col and "weighted avg" in df_metrics.index:
				try:
					f1_weighted = float(df_metrics.loc["weighted avg", f1_col])
				except Exception:
					f1_weighted = None
			if f1_weighted is not None:
				# Valor sin etiqueta (el título está arriba)
				st.metric(label="", value=f"{f1_weighted:.3f}")
			else:
				st.info("No se encontró el F1-Score (Weighted Avg) en metrics.csv")

	with col_roc:
		# Título estilizado (blanco y un poco más grande)
		st.markdown(
			"<h4 style='color:#FFFFFF; margin:0 0 0.5rem 0;'>Curva ROC</h4>",
			unsafe_allow_html=True
		)
		if not ROC_JSON.exists():
			st.error(f"No se encontró '{ROC_JSON}'.")
		else:
			with open(ROC_JSON, "r", encoding="utf-8") as f:
				roc_data = json.load(f)
			fpr = roc_data.get("fpr", [])
			tpr = roc_data.get("tpr", [])
			auc_val = roc_data.get("auc", None)

			fig_roc = go.Figure()
			fig_roc.add_trace(
				go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC", line=dict(color="#4CAF50", width=3))
			)
			fig_roc.add_trace(
				go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Azar", line=dict(color="#9E9E9E", dash="dash"))
			)
			title_txt = "Curva ROC"
			if auc_val is not None:
				try:
					title_txt += f" — AUC={float(auc_val):.3f}"
				except Exception:
					pass

			fig_roc.update_layout(
				title=title_txt,
				xaxis_title="Tasa de Falsos Positivos",
				yaxis_title="Tasa de Verdaderos Positivos",
				template="plotly_white",
				legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
			)
			st.plotly_chart(fig_roc, use_container_width=True)
