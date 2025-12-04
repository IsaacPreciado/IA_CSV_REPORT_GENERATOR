
# CREAR EL ENTORNO VIRTUAL (en la carpeta raiz del proyecto)

python -m venv venv

# ACTIVAR EL ENTORNO

Si usas Windows (CMD / PowerShell):

.\venv\Scripts\activate

Si usas Linux / macOS:

source venv/bin/activate

# INSTALAR DEPENDENCIAS

Asegúrate de que el archivo 'requirements.txt' esté actualizado.

pip install -r requirements.txt

# GEMINI_API_KEY:

Asegúrate de configurar tu api key en el archivo .env, de lo contrario, no se podrá generar el reporte con IA.

# EJECUTAR LA APLICACIÓN

streamlit run app.py

# DESACTIVAR (Al terminar de trabajar)

deactivate
