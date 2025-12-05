"""
Módulo de Análisis Estadístico.

Este módulo proporciona la clase StatsAnalyzer, que se encarga de realizar
cálculos estadísticos clave sobre un DataFrame de pandas.
"""
import numpy as np

class StatsAnalyzer:
    """
    Realiza una serie de análisis estadísticos sobre un DataFrame.

    Esta clase toma un DataFrame y listas de columnas numéricas y categóricas
    para calcular métricas como el porcentaje de valores faltantes, las modas
    de las columnas categóricas, las correlaciones entre variables numéricas
    y la cantidad de valores atípicos (outliers).

    Attributes:
        df (pd.DataFrame): El DataFrame a analizar.
        numerical_cols (list): Lista de nombres de columnas numéricas.
        categorical_cols (list): Lista de nombres de columnas categóricas.
    """
    def __init__(self, df, numerical_cols, categorical_cols):
        """
        Inicializa el StatsAnalyzer.

        Args:
            df (pd.DataFrame): El DataFrame de pandas a analizar.
            numerical_cols (list): Lista de nombres de las columnas numéricas.
            categorical_cols (list): Lista de nombres de las columnas categóricas.
        """
        self.df = df
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols

    def get_missing_percentage(self):
        """
        Calcula el porcentaje de valores nulos para cada columna.

        Returns:
            pd.Series: Una serie con el porcentaje de nulos para las columnas
                       que tienen al menos un valor faltante, ordenado de
                       mayor a menor.
        """
        missing = self.df.isnull().mean() * 100
        return missing[missing > 0].sort_values(ascending=False)

    def get_categorical_modes(self):
        """
        Obtiene la moda (valor más frecuente) de las columnas categóricas.

        Ignora columnas con alta cardinalidad (más de 50 valores únicos) para
        evitar calcular la moda en columnas de texto libre o identificadores.

        Returns:
            dict: Un diccionario donde las claves son los nombres de las columnas
                  y los valores son sus respectivas modas.
        """
        modes = {}
        for col in self.categorical_cols:
            # Heurística para ignorar columnas que parecen texto libre o IDs.
            if self.df[col].nunique() > 50: continue
            modes[col] = self.df[col].mode()[0]
        return modes

    def calculate_correlations(self):
        """
        Calcula la matriz de correlación de Pearson y extrae las más altas.

        Si hay menos de dos columnas numéricas, no se puede calcular la correlación.
        Extrae las 5 correlaciones más fuertes (en valor absoluto) sin duplicados.

        Returns:
            tuple: Una tupla conteniendo:
                - dict: Un diccionario con las 5 correlaciones más altas.
                - pd.DataFrame or None: La matriz de correlación completa, o None
                  si no se pudo calcular.
        """
        if len(self.numerical_cols) < 2:
            return {}, None

        # Calcula la matriz de correlación de Pearson.
        corr_matrix = self.df[self.numerical_cols].corr(method='pearson')
        
        # Crea una máscara para eliminar la diagonal superior y duplicados.
        mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        corrs = corr_matrix.where(mask).stack()
        
        # Ordena por magnitud absoluta para encontrar las más fuertes, pero conserva el signo.
        top_corr = corrs.iloc[np.argsort(-corrs.abs())].head(5)
        
        # Formatea el resultado en un diccionario para el reporte de IA.
        top_dict = {f"{i[0]} vs {i[1]}": round(v, 4) for i, v in top_corr.items()}
        return top_dict, corr_matrix

    def count_outliers_iqr(self):
        """
        Cuenta el número total de outliers en las columnas numéricas usando el método IQR.

        Un valor es considerado un outlier si está por debajo de Q1 - 1.5*IQR o
        por encima de Q3 + 1.5*IQR.

        Returns:
            int: El número total de outliers detectados en todo el dataset numérico.
        """
        total_outliers = 0
        for col in self.numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            currentOutliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
            total_outliers += currentOutliers
        return total_outliers