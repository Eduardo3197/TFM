import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1) Primer comando Streamlit: configuración de la página
st.set_page_config(page_title="Predicción de Ventas", layout="centered")

# 2) Paths relativos
BASE_DIR   = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 3) Cache manual de artefactos
_models_cache = None
def load_models():
    global _models_cache
    if _models_cache is None:
        _models_cache = {
            "xgb_general":          joblib.load(os.path.join(MODELS_DIR, "xgb_general.pkl")),
            "reliability_general":  joblib.load(os.path.join(MODELS_DIR, "general_reliability.pkl")),
            "reliability_by_store": joblib.load(os.path.join(MODELS_DIR, "reliability_by_store.pkl")),
            "predictions_by_store": joblib.load(os.path.join(MODELS_DIR, "predictions_by_store.pkl")),
            "predictions_by_dept":  joblib.load(os.path.join(MODELS_DIR, "predictions_by_dept.pkl")),
            "metrics_by_dept":      joblib.load(os.path.join(MODELS_DIR, "metrics_by_dept.pkl")),
            "dept2cluster":         joblib.load(os.path.join(MODELS_DIR, "dept2cluster.pkl"))
        }
    return _models_cache

# 4) Carga de artefactos
models = load_models()
xgb_general          = models["xgb_general"]
reliability_general  = models["reliability_general"]
reliability_by_store = models["reliability_by_store"]
predictions_by_store = models["predictions_by_store"]
predictions_by_dept  = models["predictions_by_dept"]
metrics_by_dept      = models["metrics_by_dept"]
dept2cluster         = models["dept2cluster"]

# 5) Precomputar predicciones generales agregadas (sumar por fecha)
_general_preds = None
def load_general_predictions():
    global _general_preds
    if _general_preds is None:
        df_list = []
        for df in predictions_by_store.values():
            df_list.append(df[['Date','Predicted']])
        all_df = pd.concat(df_list)
        _general_preds = all_df.groupby('Date', as_index=False)['Predicted'].sum()
    return _general_preds

general_preds = load_general_predictions()

# 6) Construir la UI
st.title("🔮 Predicción de Ventas Semanales")
opcion = st.radio(
    "Selecciona el tipo de predicción:",
    ["General", "Por Tienda", "Por Departamento"]
)

# 7) Predicción General + alertas
if opcion == "General":
    st.subheader("📦 Predicción General de Ventas")
    fechas = pd.to_datetime(general_preds['Date']).dt.date.unique().tolist()
    sel_date = st.selectbox("Selecciona la fecha (viernes)", fechas)
    df_sel = general_preds[pd.to_datetime(general_preds['Date']).dt.date == sel_date]
    if df_sel.empty:
        st.error("No hay predicción para la fecha seleccionada.")
    else:
        pred    = df_sel['Predicted'].iloc[0]
        confiab = reliability_general
        st.metric(label="Ventas Predichas (General)", value=f"${pred:,.0f}")
        st.caption(f"Confiabilidad estimada: {confiab*100:.1f}%")

        # --- ALERTAS: media ± std histórica ---
        μ = general_preds['Predicted'].mean()
        σ = general_preds['Predicted'].std()
        if pred > μ + σ:
            st.warning("⚠️ Se prevé un pico de ventas. Incrementar inventario.")
        elif pred < μ - σ:
            st.info("ℹ️ Demanda baja anticipada. Reducir pedidos.")

# 8) Predicción por Tienda + alertas
elif opcion == "Por Tienda":
    st.subheader("🏬 Predicción por Tienda")
    store_id = st.selectbox("Selecciona la tienda", sorted(predictions_by_store.keys()))
    df_store = predictions_by_store[store_id]
    fechas   = pd.to_datetime(df_store['Date']).dt.date.unique().tolist()
    sel_date = st.selectbox("Selecciona la fecha (viernes)", fechas)
    df_sel   = df_store[pd.to_datetime(df_store['Date']).dt.date == sel_date]
    if df_sel.empty:
        st.error("No hay predicción para esa tienda y fecha.")
    else:
        pred    = df_sel['Predicted'].iloc[0]
        confiab = reliability_by_store.get(store_id)
        st.metric(label=f"Tienda {store_id} ({sel_date})", value=f"${pred:,.0f}")
        if confiab is not None:
            st.caption(f"Confiabilidad (R²): {confiab:.2%}")

        # --- ALERTAS: media ± std de esa tienda ---
        μ = df_store['Predicted'].mean()
        σ = df_store['Predicted'].std()
        if pred > μ + σ:
            st.warning("⚠️ Pico de demanda en esta tienda. Incrementar inventario local.")
        elif pred < μ - σ:
            st.info("ℹ️ Baja demanda en esta tienda. Reducir pedidos.")

# 9) Predicción por Departamento + alertas
elif opcion == "Por Departamento":
    st.subheader("🧩 Predicción por Departamento")
    depts   = sorted(predictions_by_dept.keys())
    dept_id = st.selectbox("Selecciona el departamento", depts)
    df_dept = predictions_by_dept[dept_id]
    fechas  = pd.to_datetime(df_dept['Date']).dt.date.unique().tolist()
    sel_date= st.selectbox("Selecciona la fecha (viernes)", fechas)
    df_sel  = df_dept[pd.to_datetime(df_dept['Date']).dt.date == sel_date]
    if df_sel.empty:
        st.error("No hay predicción para ese departamento y fecha.")
    else:
        pred    = df_sel['Predicted'].iloc[0]
        confiab = metrics_by_dept.get(dept_id, {}).get('R2')
        st.metric(label=f"Dept. {dept_id} ({sel_date})", value=f"${pred:,.0f}")
        if confiab is not None:
            st.caption(f"Confiabilidad (R²): {confiab:.2%}")

        # --- ALERTAS: media ± std de ese departamento ---
        μ = df_dept['Predicted'].mean()
        σ = df_dept['Predicted'].std()
        if pred > μ + σ:
            st.warning("⚠️ Pico de demanda en este departamento. Aumentar inventario.")
        elif pred < μ - σ:
            st.info("ℹ️ Baja demanda en este departamento. Reducir pedidos.")
