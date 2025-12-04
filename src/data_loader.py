import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.numerical_cols = []
        self.categorical_cols = []

    def load_and_validate(self):
        """Carga el CSV, valida dimensiones y filtra columnas ID."""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"✅ Dataset cargado: {self.df.shape}")
            
            # 1. Validación de Requisitos (PDF)
            if self.df.shape[0] < 2000 or self.df.shape[1] < 10:
                print(f"⚠️ AVISO: El dataset no cumple los requisitos mínimos (2000 filas, 10 cols).")

            # 2. Clasificación y Filtrado Inteligente
            self._filter_columns()
            
            return True
        except Exception as e:
            print(f"❌ Error al cargar datos: {e}")
            return False

    def _filter_columns(self):
        """Separa columnas y elimina IDs o columnas con alta cardinalidad inútil."""
        all_num = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        self.numerical_cols = []

        print("🔍 Analizando tipos de columnas...")
        for col in all_num:
            unique_count = self.df[col].nunique()
            total_rows = len(self.df)
            is_integer = pd.api.types.is_integer_dtype(self.df[col])
            
            # Lógica Anti-ID:
            # Si es entero y todos son únicos -> Es un ID.
            if unique_count == total_rows and is_integer:
                print(f"   🚫 Ignorando '{col}': ID detectado (Entero único).")
                continue
            
            # Si el nombre sugiere ID y tiene variabilidad extrema
            is_id_name = any(x in col.lower() for x in ['id', 'code', 'codigo', 'index'])
            if is_id_name and unique_count > total_rows * 0.9:
                print(f"   🚫 Ignorando '{col}': Nombre sugiere ID.")
                continue
                
            self.numerical_cols.append(col)

        print(f"   Variables Numéricas Válidas: {len(self.numerical_cols)}")
        print(f"   Variables Categóricas Detectadas: {len(self.categorical_cols)}")

    def get_data(self):
        return self.df, self.numerical_cols, self.categorical_cols