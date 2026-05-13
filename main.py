import streamlit as st
import pandas as pd
import joblib
import traceback

# 1. Configuración de la página
st.set_page_config(page_title="Predictor Nursery", page_icon="👶", layout="wide")
st.title("👶 Sistema de Admisión: Nursery Dataset")

# 2. Carga segura del modelo
try:
    modelo = joblib.load('modelo_nursery.pkl')
    columnas = joblib.load('columnas_nursery.pkl')
    modelo_cargado = True
except Exception as e:
    st.error("❌ Error al cargar los archivos del modelo")
    with st.expander("🔍 Detalles técnicos"):
        st.code(traceback.format_exc())
    modelo_cargado = False

# 3. Construcción de la interfaz si el modelo cargó bien
if modelo_cargado:
    # Creamos las dos pestañas requeridas
    tab1, tab2 = st.tabs(["🔮 Predictor Interactivo", "📊 Resultados del Modelo"])

    # --- PESTAÑA 1: PREDICCIÓN ---
    with tab1:
        st.header("Ingreso de Datos")
        st.markdown("Seleccione las características para obtener una recomendación en tiempo real.")
        
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
            datos_usuario = pd.DataFrame([[parents, has_nurs, form, children, housing, finance, social, health]], 
                                         columns=["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health"])
            datos_encoded = pd.get_dummies(datos_usuario).reindex(columns=columnas, fill_value=0)
            prediccion = modelo.predict(datos_encoded)[0]
            st.success(f"### 🎯 Recomendación del Modelo: **{prediccion.upper()}**")

    # --- PESTAÑA 2: VISUALIZACIÓN DE RESULTADOS ---
    with tab2:
        st.header("Métricas de Desempeño (Solemne 1)")
        st.markdown("Resultados obtenidos tras el entrenamiento del Árbol de Decisión.")
        
        metricas = {
            "Clase": ["not_recom", "priority", "spec_prior", "very_recom"],
            "Precision": ["1.00", "0.96", "0.96", "0.87"],
            "Recall": ["1.00", "0.95", "0.96", "0.98"],
            "F1-Score": ["1.00", "0.95", "0.96", "0.92"],
            "Soporte": ["1296", "1280", "1213", "99"]
        }
        
        st.table(pd.DataFrame(metricas))
        st.info("💡 **Análisis:** El modelo demuestra una alta eficacia para detectar casos críticos, logrando un Recall de 0.98 en la clase minoritaria.")
