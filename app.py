import streamlit as st
import pandas as pd
import joblib
import os

# --- Columnas esperadas ---
FEATURE_COLUMNS = [
    'EDAD',
    'PUNTAJE TEST',
    'Grado G.cronica',
    'Act. Inflamatoria',
    'Grado Act. Inflamatoria',
    'Atrofia.1',
    'Daño mucinoso',
    'Grado de Daño mucinosos',
    'Extension de Daño Mucinoso',
    'Tipo de Metaplasia',
    'Metaplasia Porcentaje',
    'Numero de Foliculos linfoides'
]

# --- Configuración de la app ---
st.set_page_config(page_title="Centinela", layout="wide")
st.title("Pruebas de Predicción")

st.sidebar.header("Configuración")

# --- Selección de modelo ---
modelo_opcion = st.sidebar.selectbox(
    "Selecciona el modelo:",
    ["./models/rfe5_model.joblib", "./models/rfe8_model.joblib", "./models/rfe12_model.joblib"]
)

# Cargar modelo
if os.path.exists(modelo_opcion):
    model = joblib.load(modelo_opcion)
    st.sidebar.success(f"Modelo {modelo_opcion} cargado correctamente")
else:
    st.sidebar.error(f"No se encontró el archivo {modelo_opcion}. Sube el modelo al repositorio.")
    model = None

# --- Selección de entrada de datos ---
input_option = st.radio(
    "Selecciona el modo de ingreso de datos:",
    ("Subir CSV", "Ingresar manualmente")
)

data = None

if input_option == "Subir CSV":
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Usar solo las columnas necesarias
        missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_cols:
            st.error(f"El archivo no contiene las siguientes columnas necesarias: {missing_cols}")
        else:
            data = df[FEATURE_COLUMNS]
            st.write("Datos cargados (primeras filas):")
            st.dataframe(data.head())

elif input_option == "Ingresar manualmente":
    st.write("Introduce los valores de entrada:")
    manual_input = {}
    for col in FEATURE_COLUMNS:
        manual_input[col] = st.number_input(f"{col}", value=0.0)
    data = pd.DataFrame([manual_input])

# --- Botón de predicción ---
if st.button("🔮 Ejecutar modelo"):
    if model is None:
        st.error("Primero carga un modelo válido.")
    elif data is None:
        st.error("Debes ingresar datos antes de predecir.")
    else:
        prediction = model.predict(data)
        st.success(f"Resultado de la predicción: **{int(prediction[0])}** (columna target)")
