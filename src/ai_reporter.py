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

    # Agregamos 'cat_modes' a los argumentos
    def generate_report(self, dataset_name, shape, missing_percent, outliers, top_corr, cat_modes):
        if not self.model or not self.api_key:
            return "Error: Falta API Key o configuración del modelo.", None

        print(f"🧠 Generando Reporte con {self.model_name}...")
        
        prompt = f"""
        Actúa como Data Scientist Senior. Analiza el dataset '{dataset_name}'.
        
        METADATOS ESTADÍSTICOS:
        - Dimensiones: {shape}
        - % Nulos Promedio: {missing_percent:.2f}%
        - Outliers detectados (IQR): {outliers}
        - Top Correlaciones (Pearson con signo): {top_corr}
        
        ESTRUCTURA DEL REPORTE (Markdown):
        Resumen Ejecutivo (Estado de salud de los datos).
        3 Hallazgos Clave (Interpreta correlaciones, outliers y modas de negocio).
        3 Recomendaciones de Limpieza y Preprocesamiento.
        """

        try:
            response = self.model.generate_content(prompt)
            report_text = response.text
            
            output_folder = "reports"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            clean_name = dataset_name.split(".")[0]
            filename = f"reporte_{clean_name}.md"
            
            output_path = os.path.join(output_folder, filename)
            
            with open(output_path, "w", encoding='utf-8') as f:
                f.write(report_text)
            
            return report_text, output_path
                
        except Exception as e:
            return f"❌ Error API Gemini: {str(e)}", None