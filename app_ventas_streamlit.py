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
            "df_co":      joblib.load(os.path.join(MODELS_DIR, "df_co.pkl")),
            "dept2cluster":         joblib.load(os.path.join(MODELS_DIR, "dept2cluster.pkl"))
        }
    return _models_cache

# 4) Carga de artefactos
models = load_models()
df_co         = models["df_co"]
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
    ["General", "Por Tienda", "Por Departamento", "Predicción Futura"]
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

elif opcion == "Predicción Futura":
    st.subheader("📅 Predicción Futura Personalizada")

    tipo_futuro = st.selectbox("Selecciona el tipo de predicción futura", ["General", "Por Tienda", "Por Departamento"])

    # Generar 4 fechas futuras desde la última fecha del dataset
    future_dates = pd.date_range(start=df_co['Date'].max() + pd.Timedelta(weeks=1), periods=4, freq="W-FRI")
    sel_date = st.selectbox("Selecciona la fecha (viernes)", future_dates.strftime("%Y-%m-%d"))

    # Crear base de datos para esa fecha
    df_future = pd.DataFrame({'Date': [pd.to_datetime(sel_date)]})
    df_future['week'] = df_future['Date'].dt.isocalendar().week
    df_future['month'] = df_future['Date'].dt.month
    df_future['WeekSin'] = np.sin(2 * np.pi * df_future['week'] / 52)
    df_future['Trend'] = len(df_co) + 1
    df_future['IsHoliday'] = df_future['Date'].dt.strftime("%Y-%m-%d").isin(["2023-11-24", "2023-12-25"])
    df_future['HolidaySeason'] = df_future['month'].isin([11, 12])
    df_future['IsEndOfYear'] = (df_future['month'] == 12) & (df_future['week'] >= 50)

    # Crear PromoStrength si no existe
    if 'PromoStrength' not in df_co.columns:
        markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
        for col in markdown_cols:
            if col not in df_co.columns:
                df_co[col] = 0
        df_co[markdown_cols] = df_co[markdown_cols].fillna(0)
        df_co['PromoStrength'] = df_co[markdown_cols].sum(axis=1)

    # Variables comunes
    df_future['CPI'] = df_co['CPI'].mean()
    df_future['Unemployment'] = df_co['Unemployment'].mean()
    df_future['Fuel_Price'] = df_co['Fuel_Price'].mean()
    df_future['Temperature'] = df_co['Temperature'].mean()
    df_future['PromoStrength'] = df_co['PromoStrength'].mean()
    df_future['PromoStrength_log'] = np.log1p(df_future['PromoStrength'])
    df_future['HasPromotion'] = df_future['PromoStrength'] > 0

    # General
    if tipo_futuro == "General":
        df_future['Size'] = df_co['Size'].mean()
        df_future['Type_B'] = 0
        df_future['Type_C'] = 0
        features = ['Size', 'CPI', 'Unemployment', 'Fuel_Price',
                    'IsHoliday', 'HasPromotion', 'PromoStrength',
                    'HolidaySeason', 'IsEndOfYear', 'week', 'month',
                    'WeekSin', 'Trend', 'Type_B', 'Type_C']
        pred = xgb_general.predict(df_future[features])[0]

    # Por Tienda
    elif tipo_futuro == "Por Tienda":
        tienda = st.selectbox("Selecciona la tienda", sorted(predictions_by_store.keys()))
        clust = 0  
        modelo = xgb_general  
        tienda_df = df_co[df_co['Store'] == tienda]
        df_future['Size'] = tienda_df['Size'].iloc[0] if not tienda_df.empty else df_co['Size'].mean()
        df_future['Type_B'] = 0
        df_future['Type_C'] = 0
        features = ['Size', 'CPI', 'Unemployment', 'Fuel_Price',
                    'IsHoliday', 'HasPromotion', 'PromoStrength',
                    'HolidaySeason', 'IsEndOfYear', 'week', 'month',
                    'WeekSin', 'Trend', 'Type_B', 'Type_C']
        pred = modelo.predict(df_future[features])[0]

    # Por Departamento
    elif tipo_futuro == "Por Departamento":
        dept = st.selectbox("Selecciona el departamento", sorted(predictions_by_dept.keys()))
        clust = dept2cluster.get(dept, 0)
        modelo = xgb_general  
        features = ['Size', 'CPI', 'Unemployment', 'Fuel_Price',
                    'IsHoliday', 'HasPromotion', 'PromoStrength',
                    'HolidaySeason', 'IsEndOfYear', 'week', 'month',
                    'WeekSin', 'Trend', 'Type_B', 'Type_C']
        df_future['Size'] = df_co[df_co['Dept'] == dept]['Size'].mean() if dept in df_co['Dept'].values else df_co['Size'].mean()
        df_future['Type_B'] = 0
        df_future['Type_C'] = 0
        pred = modelo.predict(df_future[features])[0]

    st.metric(f"Predicción para {sel_date}", value=f"${pred:,.0f}")

    # --- Alerta basada en histórico general ---
    μ = general_preds['Predicted'].mean()
    σ = general_preds['Predicted'].std()

    if pred > μ + σ:
        st.warning("🔺 Se prevé un pico de ventas. Incrementar inventario.")
    elif pred < μ - σ:
        st.info("🔵 Demanda baja anticipada. Reducir pedidos.")
