"""
Archivo principal de la aplicación Streamlit "Auto Data Analyst AI".

Este script construye la interfaz de usuario, maneja la carga de archivos,
orquesta el análisis de datos (estadísticas y visualizaciones) y
genera un reporte utilizando un modelo de IA (Gemini).
"""
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Importar nuestros módulos
from src.data_loader import DataLoader
from src.statistics import StatsAnalyzer
from src.visualization import Visualizer
from src.ai_reporter import AIReporter

# Configuración de página
st.set_page_config(page_title="Auto Data Analyst AI", layout="wide", page_icon="📊")
# Carga las variables de entorno desde un archivo .env si existe.
load_dotenv()

# --- SIDEBAR: Configuración ---
st.sidebar.header("⚙️ Configuración")

# 1. Selector de API Key de Gemini
default_api_key = os.getenv("GEMINI_API_KEY")
user_api_key = st.sidebar.text_input(
    "Gemini API Key", 
    value=default_api_key if default_api_key else "", 
    type="password",
    help="Se cargó automáticamente del .env si existe. Puedes sobreescribirla."
)

# 2. Selector del Modelo de IA
model_options = [
    "gemini-2.5-flash", 
    "gemini-2.0-flash-exp", 
    "gemini-1.5-flash", 
    "gemini-1.5-pro"
]
selected_model = st.sidebar.selectbox("Selecciona el Modelo", model_options)

# --- SIDEBAR: Información ---
st.sidebar.markdown("---")
st.sidebar.info("Sube un CSV para comenzar el análisis automático.")

# --- MAIN AREA ---
st.title("📊 Auto Data Analyst con IA")
st.markdown("Carga tu dataset, visualiza estadísticas y obtén un reporte profesional generado por Gemini.")

# --- 1. Carga de Archivo ---
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

# Si se ha subido un archivo, comienza el proceso de análisis.
if uploaded_file is not None:
    # Se guarda el archivo temporalmente para poder ser leído por pandas.
    temp_filename = uploaded_file.name
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Bloque principal de ejecución con manejo de errores.
    try:
        # --- A. Carga y Validación ---
        with st.spinner('Cargando y validando datos...'):
            loader = DataLoader(temp_filename)
            if loader.load_and_validate():
                st.success(f"✅ Archivo cargado exitosamente: {uploaded_file.name}")
                df, num_cols, cat_cols = loader.get_data()
            else:
                # Detiene la ejecución si el archivo no cumple con el tamaño mínimo.
                st.error("❌ El archivo no cumple con los requisitos mínimos (Filas < 2000 o Cols < 10).")
                st.stop()

        # --- B. Análisis Estadístico ---
        stats = StatsAnalyzer(df, num_cols, cat_cols) # Instancia del analizador
        
        # Cálculos
        missing_series = stats.get_missing_percentage()
        missing_avg = missing_series.mean() if not missing_series.empty else 0
        top_corrs, corr_matrix = stats.calculate_correlations()
        total_outliers = stats.count_outliers_iqr()
        cat_modes = stats.get_categorical_modes()

        # Mostrar KPIs (Key Performance Indicators) Generales del Dataset
        st.markdown("### 📈 Resumen General")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Filas", f"{df.shape[0]:,}")
        kpi2.metric("Columnas", df.shape[1])
        kpi3.metric("% Nulos Promedio", f"{missing_avg:.2f}%")
        kpi4.metric("Total Outliers", f"{total_outliers:,}")
        
        st.divider()

        # Mostrar las modas (valores más frecuentes) de las columnas categóricas.
        st.markdown("### 🏆 Modas (Valores más frecuentes)")
        
        if cat_modes:
            with st.expander("Ver detalle de categorías", expanded=True):
                # Se crea un grid dinámico para mostrar las métricas de moda.
                # Calculamos cuántas columnas usar en el grid (ej. 3 por fila)
                cols = st.columns(3)
                for idx, (col_name, mode_val) in enumerate(cat_modes.items()):
                    # Usamos el operador módulo para ciclar entre las 3 columnas
                    with cols[idx % 3]:
                        st.metric(
                            label=col_name, 
                            value=str(mode_val), 
                            delta="Moda", 
                            delta_color="off" # off = gris neutro
                        )
        else:
            st.info("No hay variables categóricas relevantes para calcular moda.")

        st.divider()

        # --- C. Visualización de Datos ---
        st.subheader("🎨 Visualizaciones Automáticas")
        viz = Visualizer() # Instancia del visualizador
        
        # Creación de pestañas para organizar los gráficos.
        tab1, tab2, tab3 = st.tabs(["Mapa de Calor & Correlación", "Distribuciones Numéricas", "Frecuencia Categórica"])
        
        # Pestaña 1: Mapas de calor para valores nulos y correlaciones.
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
                    st.warning("No hay suficientes datos numéricos.")

        # Pestaña 2: Histogramas y Boxplots para cada variable numérica.
        with tab2:
            st.markdown("**Top Variables Numéricas**")
            for col in num_cols:
                st.pyplot(viz.create_numerical_distributions(df, col))
        
        # Pestaña 3: Gráficos de barras para variables categóricas con baja cardinalidad.
        with tab3:
            st.markdown("**Top Variables Categóricas**")
            valid_cats = [c for c in cat_cols if df[c].nunique() <= 20]
            if valid_cats:
                for col in valid_cats:
                    st.pyplot(viz.create_categorical_count(df, col))
            else:
                st.info("No hay variables aptas para graficar.")

        st.divider()

        # --- D. Generación de Reporte con IA ---
        st.subheader("🧠 Reporte Inteligente")
        
        col_gen, col_info = st.columns([1, 2])
        
        with col_gen:
            generate_btn = st.button("Generar Insights con Gemini", type="primary", use_container_width=True)
        
        # Si el usuario hace clic en el botón de generar.
        if generate_btn:
            # Validar que la API Key ha sido introducida.
            if not user_api_key:
                st.error("⚠️ Falta la API Key. Configúrala en el menú lateral.")
            else:
                with st.spinner(f"Analizando datos con {selected_model}..."):
                    # Instanciar y llamar al generador de reportes.
                    reporter = AIReporter(user_api_key, selected_model)
                    report_content, saved_path = reporter.generate_report(
                        dataset_name=uploaded_file.name,
                        shape=df.shape,
                        missing_percent=missing_avg,
                        outliers=total_outliers,
                        top_corr=top_corrs,
                        cat_modes=cat_modes
                    )
                    
                    # Si el reporte se genera correctamente.
                    if saved_path:
                        st.success("¡Análisis completado!")
                        # Mostrar el contenido del reporte en la app.
                        with st.container(border=True):
                            st.markdown(report_content)
                        # Ofrecer la opción de descargar el reporte.
                        st.download_button(
                            label="📥 Descargar Reporte (Markdown)",
                            data=report_content,
                            file_name=os.path.basename(saved_path),
                            mime="text/markdown"
                        )
                    else:
                        # Mostrar el mensaje de error si la generación falla.
                        st.error(report_content)

    except Exception as e:
        # Captura cualquier error inesperado durante el proceso.
        st.error(f"Error inesperado: {e}")
    
    finally:
        # Asegura que el archivo temporal se elimine al final, incluso si hay errores.
        if os.path.exists(temp_filename):
            os.remove(temp_filename)