import streamlit as st
import pandas as pd
import joblib
import traceback

# 1. Configuración de la página (Ahora con layout más amplio)
st.set_page_config(page_title="Nursery Predictor", page_icon="🏢", layout="wide")

# --- BARRA LATERAL (SIDEBAR) ESTILO CORPORATIVO ---
with st.sidebar:
    st.title("⚙️ Panel de Control")
    st.markdown("---")
    st.info("Sistema automatizado de evaluación de perfiles para asignación de cupos.")
    st.markdown("Desarrollado para la Solemne 2.")
    st.markdown("---")
    st.caption("Motor de Inferencia: Árbol de Decisión V1.0")

# 2. Título principal con estilo
st.title("🏢 Sistema de Admisión: Nursery Dataset")
st.markdown("*Clasificación inteligente de solicitudes mediante Machine Learning*")

# 3. Carga segura del modelo
try:
    modelo = joblib.load('modelo_nursery.pkl')
    columnas = joblib.load('columnas_nursery.pkl')
    modelo_cargado = True
except Exception as e:
    st.error("❌ Error de conexión con el modelo.")
    with st.expander("Ver consola de errores"):
        st.code(traceback.format_exc())
    modelo_cargado = False

# 4. Construcción de la interfaz
if modelo_cargado:
    tab1, tab2 = st.tabs(["🔮 Predictor de Solicitudes", "📊 Análisis de Rendimiento"])

    # --- PESTAÑA 1: PREDICCIÓN MEJORADA ---
    with tab1:
        st.markdown("### Formulario de Evaluación Familiar")
        
        # Agrupamos las opciones en tarjetas visuales (contenedores)
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                parents = st.selectbox("Perfil de los Padres", ["usual", "pretentious", "great_pret"])
                children = st.selectbox("Número de hijos", ["1", "2", "3", "more"])
                form = st.selectbox("Estructura familiar", ["complete", "completed", "incomplete", "foster"])
            with col2:
                housing = st.selectbox("Condiciones de Vivienda", ["convenient", "less_conv", "critical"])
                finance = st.selectbox("Situación Financiera", ["convenient", "inconv"])
            with col3:
                social = st.selectbox("Entorno Social", ["nonprob", "slightly_prob", "problematic"])
                health = st.selectbox("Estado de Salud", ["recommended", "priority", "not_recom"])
                has_nurs = st.selectbox("Necesidad de cuidado", ["proper", "less_proper", "improper", "critical", "very_crit"])

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Botón de acción destacado
        if st.button("Ejecutar Análisis de Perfil", type="primary", use_container_width=True):
            datos_usuario = pd.DataFrame([[parents, has_nurs, form, children, housing, finance, social, health]], 
                                         columns=["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health"])
            datos_encoded = pd.get_dummies(datos_usuario).reindex(columns=columnas, fill_value=0)
            prediccion = modelo.predict(datos_encoded)[0]
            
            # --- RESPUESTA DINÁMICA CON COLORES ---
            st.markdown("---")
            if prediccion == "very_recom" or prediccion == "recommended":
                st.success(f"### 🟢 Decisión Sugerida: **{prediccion.upper()}**")
                st.balloons() # ¡Un toque de celebración!
            elif prediccion == "priority" or prediccion == "spec_prior":
                st.warning(f"### 🟡 Decisión Sugerida: **{prediccion.upper()}**")
            else:
                st.error(f"### 🔴 Decisión Sugerida: **{prediccion.upper()}**")

    # --- PESTAÑA 2: DASHBOARD DE MÉTRICAS ---
    with tab2:
        st.markdown("### Resumen Ejecutivo del Modelo")
        
        # Destacamos el logro principal con un componente visual de métricas
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric(label="Precisión Global", value="94%")
        col_m2.metric(label="Recall (Casos Críticos)", value="98%", delta="Alta sensibilidad")
        col_m3.metric(label="Penalización", value="Activada", delta="Balance de clases", delta_color="off")
        
        st.markdown("---")
        st.markdown("**Matriz de Validación (Conjunto de Test)**")
        
        metricas = {
            "Clasificación": ["Not Recommended", "Priority", "Special Priority", "Very Recommended"],
            "Precision": ["1.00", "0.96", "0.96", "0.87"],
            "Recall": ["1.00", "0.95", "0.96", "0.98"],
            "F1-Score": ["1.00", "0.95", "0.96", "0.92"],
            "Volumen (Soporte)": ["1296", "1280", "1213", "99"]
        }
        st.dataframe(pd.DataFrame(metricas), use_container_width=True)
