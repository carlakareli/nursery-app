import streamlit as st
import pandas as pd
import joblib
import traceback

# Configuración inicial
st.set_page_config(page_title="Predictor Nursery", page_icon="👶", layout="wide")
st.title("👶 Sistema de Admisión: Nursery Dataset")

# MODO DEBUG: Bloque ultra seguro para cargar el modelo
try:
    modelo = joblib.load('modelo_nursery.pkl')
    columnas = joblib.load('columnas_nursery.pkl')
    st.success("✅ ¡El modelo y las columnas se cargaron perfectamente!")
    modelo_cargado = True
except Exception as e:
    st.error("❌ Ocurrió un error silencioso al cargar el archivo .pkl")
    # Este es el Expander de la Fase 4 del checklist:
    with st.expander("🔍 Ver detalles técnicos del error (Traceback)"):
        st.code(traceback.format_exc())
    modelo_cargado = False

# Si pasa la prueba, mostramos la interfaz de la Solemne 2
if modelo_cargado:
    st.header("🔮 Predictor Interactivo")
    st.markdown("Seleccione las características de la familia para predecir el nivel de recomendación.")
    
    col1, col2 = st.columns(2)
    with col1:
        parents = st.selectbox("Padres (parents)", ["usual", "pretentious", "great_pret"])
        has_nurs = st.selectbox("Necesidad de cuidado (has_nurs)", ["proper", "less_proper", "improper", "critical", "very_crit"])
        form = st.selectbox("Estructura familiar (form)", ["complete", "completed", "incomplete", "foster"])
        children = st.selectbox("Número de hijos (children)", ["1", "2", "3", "more"])
        
    with col2:
        housing = st.selectbox("Vivienda (housing)", ["convenient", "less_conv", "critical"])
        finance = st.selectbox("Finanzas (finance)", ["convenient", "inconv"])
        social = st.selectbox("Entorno social (social)", ["nonprob", "slightly_prob", "problematic"])
        health = st.selectbox("Salud (health)", ["recommended", "priority", "not_recom"])

    if st.button("Generar Predicción", type="primary", use_container_width=True):
        try:
            # Empaquetar y transformar los datos
            datos_usuario = pd.DataFrame([[parents, has_nurs, form, children, housing, finance, social, health]], 
                                         columns=["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health"])
            datos_encoded = pd.get_dummies(datos_usuario).reindex(columns=columnas, fill_value=0)
            
            # Predecir
            prediccion = modelo.predict(datos_encoded)[0]
            st.success(f"### 🎯 Recomendación del Modelo: **{prediccion.upper()}**")
        except Exception as e:
            st.error("Error al procesar los datos ingresados.")
            with st.expander("🔍 Ver detalles del error de predicción"):
                st.code(traceback.format_exc())
