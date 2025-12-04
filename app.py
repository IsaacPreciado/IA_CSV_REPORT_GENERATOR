import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Importar nuestros módulos
from src.data_loader import DataLoader
from src.stats import StatsAnalyzer
from src.visualization import Visualizer
from src.ai_reporter import AIReporter

# Configuración de página
st.set_page_config(page_title="Auto Data Analyst AI", layout="wide", page_icon="📊")
load_dotenv()

# --- SIDEBAR: Configuración ---
st.sidebar.header("⚙️ Configuración")

# 1. API Key Selector
default_api_key = os.getenv("GEMINI_API_KEY")
user_api_key = st.sidebar.text_input(
    "Gemini API Key", 
    value=default_api_key if default_api_key else "", 
    type="password",
    help="Se cargó automáticamente del .env si existe. Puedes sobreescribirla."
)

# 2. Model Selector
model_options = [
    "gemini-2.5-flash", 
    "gemini-2.0-flash-exp", 
    "gemini-1.5-flash", 
    "gemini-1.5-pro"
]
selected_model = st.sidebar.selectbox("Selecciona el Modelo", model_options)

st.sidebar.markdown("---")
st.sidebar.info("Sube un CSV para comenzar el análisis automático.")

# --- MAIN AREA ---
st.title("📊 Auto Data Analyst con IA")
st.markdown("Carga tu dataset, visualiza estadísticas y obtén un reporte profesional generado por Gemini.")

# 1. Carga de Archivo
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:

    temp_filename = uploaded_file.name
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # --- A. Carga y Validación ---
        with st.spinner('Cargando y validando datos...'):
            loader = DataLoader(temp_filename)
            if loader.load_and_validate():
                st.success(f"✅ Archivo cargado exitosamente: {uploaded_file.name}")
                df, num_cols, cat_cols = loader.get_data()
            else:
                st.error("❌ El archivo no cumple con los requisitos mínimos (Filas < 2000 o Cols < 10).")
                st.stop()

        # --- B. Estadísticas ---
        stats = StatsAnalyzer(df, num_cols, cat_cols)
        missing_series = stats.get_missing_percentage()
        missing_avg = missing_series.mean() if not missing_series.empty else 0
        top_corrs, corr_matrix = stats.calculate_correlations()
        total_outliers = stats.count_outliers_iqr()

        # Métricas en columnas
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Filas", df.shape[0])
        col2.metric("Columnas", df.shape[1])
        col3.metric("% Nulos Promedio", f"{missing_avg:.2f}%")
        col4.metric("Total Outliers (IQR)", total_outliers)

        # --- C. Visualización ---
        st.subheader("🎨 Visualizaciones Automáticas")
        viz = Visualizer()
        
        tab1, tab2, tab3 = st.tabs(["Mapa de Calor & Correlación", "Distribuciones Numéricas", "Frecuencia Categórica"])
        
        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Valores Faltantes**")
                fig_miss = viz.create_missing_heatmap(df)
                st.pyplot(fig_miss)
            with col_b:
                st.markdown("**Matriz de Correlación**")
                if corr_matrix is not None:
                    fig_corr = viz.create_correlation_heatmap(corr_matrix)
                    st.pyplot(fig_corr)
                else:
                    st.info("No hay suficientes datos numéricos para correlación.")

        with tab2:
            st.markdown("**Top 3 Variables Numéricas**")
            for col in num_cols:
                st.pyplot(viz.create_numerical_distributions(df, col))
        
        with tab3:
            st.markdown("**Top 3 Variables Categóricas (Filtradas)**")
            valid_cats = [c for c in cat_cols if df[c].nunique() <= 20]
            if valid_cats:
                for col in valid_cats:
                    st.pyplot(viz.create_categorical_count(df, col))
            else:
                st.info("No se encontraron variables categóricas aptas para graficar (demasiados valores únicos).")

        # --- D. Reporte IA ---
        st.subheader("🧠 Generación de Reporte con IA")
        
        generate_btn = st.button("Generar Insights con Gemini", type="primary")
        
        if generate_btn:
            if not user_api_key:
                st.error("⚠️ Por favor ingresa una API Key válida en la barra lateral.")
            else:
                with st.spinner(f"Consultando a {selected_model}..."):
                    reporter = AIReporter(user_api_key, selected_model)
                    content, error = reporter.generate_report(
                        dataset_name=uploaded_file.name,
                        shape=df.shape,
                        missing_percent=missing_avg,
                        outliers=total_outliers,
                        top_corr=top_corrs
                    )
                    
                    if not error:
                        st.success("¡Reporte generado exitosamente!")
                        st.markdown(content)
                        
                        # Botón para descargar
                        st.download_button(
                            label="Descargar Reporte MD",
                            data=content,
                            file_name=os.path.basename(uploaded_file.name.split(".")[0] + "_report.md"),
                            mime="text/markdown"
                        )
                    else:
                        st.error(content) # Muestra el mensaje de error de la API

    except Exception as e:
        st.error(f"Ocurrió un error inesperado: {e}")
    
    finally:
        # Limpieza del archivo temporal
        if os.path.exists(temp_filename):
            os.remove(temp_filename)