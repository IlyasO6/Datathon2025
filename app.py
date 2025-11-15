
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
		--bg-main: #F8FFF4; /* fondo principal */
		--bg-card: #FFFFFF; /* contenedores */
		--text-main: #1E3923; /* texto principal */
		--primary: #4CAF50; /* verde ecológico */
		--primary-dark: #388E3C; /* hover */
	  }

	  html, body, [data-testid="stAppViewContainer"] {
		background: var(--bg-main) !important;
		color: var(--text-main) !important;
	  }

	  h1, h2, h3, h4, h5, h6 { color: var(--text-main) !important; }

	  /* Tarjetas / contenedores personalizados */
	  .card {
		background: var(--bg-card);
		border-radius: 10px;
		box-shadow: 0 4px 8px rgba(0,0,0,0.08);
		padding: 1rem 1.25rem;
		border: 1px solid rgba(0,0,0,0.04);
	  }

	  /* Botones */
	  div.stButton > button,
	  .stDownloadButton > button,
	  .st-emotion-cache-1vt4y43 { /* algunos temas usan estas clases dinámicas */
		background-color: var(--primary) !important;
		color: #fff !important;
		border: none !important;
	  }
	  div.stButton > button:hover,
	  .stDownloadButton > button:hover {
		background-color: var(--primary-dark) !important;
	  }

	  /* Select / inputs */
	  .stSelectbox div[data-baseweb="select"] > div {
		border-radius: 10px !important;
		border-color: rgba(0,0,0,0.1) !important;
	  }

	  /* Métrica destacada */
	  [data-testid="stMetricValue"] {
		color: var(--primary) !important;
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
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.title("Dashboard de Modelo")
st.write("Explora resultados, interpretabilidad y métricas del modelo.")
st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# Tabs principales
# ----------------------------
tab_user, tab_shap, tab_metrics = st.tabs([
	"Resultados (User-Friendly)",
	"Análisis Técnico (SHAP)",
	"Validez del Modelo (Métricas)",
])


# ----------------------------
# Tab 1: Resultados (User-Friendly)
# ----------------------------
with tab_user:
	st.markdown("<div class='card'>", unsafe_allow_html=True)
	st.subheader("Impacto de variables por muestra")

	missing = []
	for p in [X_TEST_CSV, SHAP_VALUES_PKL]:
		if not p.exists():
			missing.append(str(p))
	if missing:
		st.error("Faltan archivos necesarios: " + ", ".join(missing))
		st.markdown("</div>", unsafe_allow_html=True)
		st.stop()

	X_test = load_x_test()
	shap_values = load_shap_values()  # [n_samples, n_features]

	if shap_values.shape[0] != len(X_test):
		st.warning(
			f"El número de filas en shap_values ({shap_values.shape[0]}) no coincide con X_test ({len(X_test)})."
		)

	id_series = pick_sample_id_options(X_test)
	id_options = id_series.tolist()
	default_idx = 0
	selected_id = st.selectbox("Selecciona ID de muestra", options=id_options, index=default_idx)

	# Encontrar posición de la muestra
	try:
		# si hay columna id
		if "id" in X_test.columns:
			sample_pos = int(X_test.index[X_test["id"] == selected_id][0])
		else:
			sample_pos = int(selected_id)
	except Exception:
		sample_pos = 0

	sample_pos = np.clip(sample_pos, 0, len(X_test) - 1)
	x_row = X_test.iloc[sample_pos]
	sv_row = shap_values[sample_pos]

	# Top-5 por |SHAP|
	abs_vals = np.abs(sv_row)
	top_idx = np.argsort(-abs_vals)[:5]
	cols = X_test.columns[top_idx]

	st.write("Top 5 variables por impacto SHAP (por muestra seleccionada):")
	for feat in cols:
		c1, c2, c3 = st.columns([3, 2, 1])
		val = x_row[feat]
		sv = float(sv_row[X_test.columns.get_loc(feat)])
		emo = emoji_for_shap(sv)
		with c1:
			st.markdown(f"**{feat}**")
		with c2:
			st.markdown(f"Valor: `{val}`")
		with c3:
			st.markdown(f"{emo}")
		st.caption(
			f"La variable '{feat}' está {'aumentando' if sv>0 else 'disminuyendo' if sv<0 else 'sin cambiar'} ({emo}) la probabilidad de la clase positiva."
		)

	st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# Tab 2: Análisis Técnico (SHAP)
# ----------------------------
with tab_shap:
	st.markdown("<div class='card'>", unsafe_allow_html=True)
	st.subheader("Gráficos SHAP")

	missing = []
	for p in [EXPLAINER_PKL, SHAP_VALUES_PKL, X_TEST_CSV]:
		if not p.exists():
			missing.append(str(p))
	if missing:
		st.error("Faltan archivos necesarios: " + ", ".join(missing))
		st.markdown("</div>", unsafe_allow_html=True)
		st.stop()

	_ = load_explainer()  # cargado aunque no se use estrictamente para summary_plot
	X_test = load_x_test()
	shap_values = load_shap_values()

	# Beeswarm
	st.markdown("**Beeswarm**")
	fig1 = plt.figure(figsize=(10, 4))
	shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
	st.pyplot(fig1, use_container_width=True)
	plt.close(fig1)

	# Bar
	st.markdown("**Importancia (Bar Plot)**")
	fig2 = plt.figure(figsize=(10, 4))
	shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
	st.pyplot(fig2, use_container_width=True)
	plt.close(fig2)

	st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# Tab 3: Validez del Modelo (Métricas)
# ----------------------------
with tab_metrics:
	st.markdown("<div class='card'>", unsafe_allow_html=True)
	st.subheader("Métricas de Rendimiento")

	df_metrics = load_metrics_df()
	if df_metrics.empty:
		st.error("No se pudieron cargar las métricas. Verifica 'classification_report.json'.")
	else:
		# f1-score de weighted avg (manejar nombre de columna)
		f1_col = "f1-score" if "f1-score" in df_metrics.columns else ("f1_score" if "f1_score" in df_metrics.columns else None)
		f1_weighted = None
		if f1_col and "weighted avg" in df_metrics.index:
			try:
				f1_weighted = float(df_metrics.loc["weighted avg", f1_col])
			except Exception:
				f1_weighted = None
		if f1_weighted is not None:
			st.metric(label="F1-Score (Weighted Avg)", value=f"{f1_weighted:.3f}")
		else:
			st.info("No se encontró el F1-Score (Weighted Avg) en metrics.csv")

		st.write("Tabla completa de métricas")
		st.dataframe(df_metrics, use_container_width=True)

	st.divider()

	# Curva ROC
	st.subheader("Curva ROC")
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

	st.markdown("</div>", unsafe_allow_html=True)
