"""
Módulo de Carga de Datos.

Este módulo proporciona la clase DataLoader, responsable de cargar,
validar y pre-procesar un archivo CSV para el análisis.
"""
import pandas as pd
import numpy as np

class DataLoader:
    """
    Gestiona la carga y validación inicial de un dataset desde un archivo CSV.

    Esta clase lee un archivo CSV, verifica que cumpla con las dimensiones
    mínimas requeridas, y separa las columnas en numéricas y categóricas,
    filtrando aquellas que parecen ser identificadores únicos.

    Attributes:
        file_path (str): La ruta al archivo CSV a cargar.
        df (pd.DataFrame): El DataFrame de pandas cargado. None si no se ha cargado.
        numerical_cols (list): Lista de nombres de columnas numéricas válidas.
        categorical_cols (list): Lista de nombres de columnas categóricas.
    """
    def __init__(self, file_path):
        """
        Inicializa el DataLoader con la ruta al archivo.

        Args:
            file_path (str): La ruta al archivo CSV.
        """
        self.file_path = file_path
        self.df = None
        self.numerical_cols = []
        self.categorical_cols = []

    def load_and_validate(self):
        """
        Carga el archivo CSV, valida sus dimensiones y filtra las columnas.

        Intenta leer el archivo CSV especificado en `file_path`. Luego, valida
        que el DataFrame resultante tenga al menos 2000 filas y 10 columnas.
        Si la validación es exitosa, procede a clasificar y filtrar las columnas.

        Returns:
            bool: True si la carga y validación son exitosas, False en caso contrario.
        """
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Dataset cargado: {self.df.shape}")
            
            if self.df.shape[0] < 2000 or self.df.shape[1] < 10:
                return False

            self._filter_columns()
            
            return True
        except Exception as e:
            print(f"❌ Error al cargar datos: {e}")
            return False

    def _filter_columns(self):
        """
        Clasifica las columnas en numéricas y categóricas, y filtra las de tipo ID.

        Este método privado identifica las columnas numéricas y categóricas basándose
        en sus tipos de datos. Además, implementa una heurística para detectar y
        excluir columnas numéricas que probablemente son identificadores (ej. 'id', 'codigo')
        basándose en su nombre y en tener una alta cardinalidad (muchos valores únicos).
        """
        all_num = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        self.numerical_cols = []

        print("🔍 Analizando tipos de columnas...")
        for col in all_num:
            unique_count = self.df[col].nunique()
            total_rows = len(self.df)
            
            # Heurística: si el nombre sugiere ID y tiene una cardinalidad muy alta.
            is_id_name = any(x in col.lower() for x in ['id', 'code', 'codigo', 'index'])
            if is_id_name and unique_count > total_rows * 0.9:
                print(f"Ignorando '{col}': Nombre sugiere ID.")
                continue
                
            self.numerical_cols.append(col)

        print(f"   Variables Numéricas Válidas: {len(self.numerical_cols)}")
        print(f"   Variables Categóricas Detectadas: {len(self.categorical_cols)}")

    def get_data(self):
        """
        Retorna los datos procesados.

        Returns:
            tuple: Una tupla conteniendo:
                - pd.DataFrame: El DataFrame cargado.
                - list: La lista de columnas numéricas.
                - list: La lista de columnas categóricas.
        """
        return self.df, self.numerical_cols, self.categorical_cols