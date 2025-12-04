import google.generativeai as genai
import os

class AIReporter:
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        self.client = None
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
            except Exception as e:
                print(f"Error config: {e}")
        else:
            print("⚠️ AVISO: No hay API Key configurada.")

    def generate_report(self, dataset_name, shape, missing_percent, outliers, top_corr):
        if not self.model or not self.api_key:
            return "Error: Falta API Key o configuración del modelo."
        print(f"🧠 Generando Reporte con {self.model_name}...")
        
        prompt = f"""
        Actúa como Data Scientist Senior. Analiza el dataset '{dataset_name}'.
        
        METADATOS:
        - Dimensiones: {shape}
        - % Nulos Promedio: {missing_percent:.2f}%
        - Outliers detectados (IQR): {outliers}
        - Top Correlaciones (Pearson con signo): {top_corr}
        
        ESTRUCTURA DEL REPORTE (Markdown):
        1. Resumen Ejecutivo (Estado de salud de los datos).
        2. Tres Hallazgos Clave (Basado en las correlaciones y outliers).
        3. Tres Recomendaciones de Limpieza y Preprocesamiento.
        """

        try:
            response = self.model.generate_content(prompt)
            report_text = response.text
            
            return report_text, False
                
        except Exception as e:
            return str(e), True