"""
Módulo de Visualización de Datos.

Este módulo proporciona la clase Visualizer, que utiliza Matplotlib y Seaborn
para generar diversas gráficas estándar para el análisis exploratorio de datos.
"""
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    """
    Crea visualizaciones estándar para el análisis de datos.

    Esta clase encapsula la lógica para generar figuras de Matplotlib, como
    mapas de calor, histogramas, boxplots y gráficos de barras, utilizando
    un estilo consistente de Seaborn.
    """
    def __init__(self):
        """
        Inicializa el Visualizer y establece el tema global para las gráficas.
        """
        sns.set_theme(style="whitegrid")

    def create_missing_heatmap(self, df):
        """
        Crea un mapa de calor para visualizar la ubicación de valores faltantes.

        Args:
            df (pd.DataFrame): El DataFrame a visualizar.

        Returns:
            matplotlib.figure.Figure: La figura de Matplotlib con el mapa de calor.
        """
        fig = plt.figure(figsize=(10, 8.5))
        sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
        plt.title('Mapa de Valores Faltantes')
        plt.tight_layout()
        return fig

    def create_numerical_distributions(self, df, col):
        """
        Crea un histograma y un boxplot para una columna numérica.

        Args:
            df (pd.DataFrame): El DataFrame que contiene los datos.
            col (str): El nombre de la columna numérica a visualizar.

        Returns:
            matplotlib.figure.Figure: La figura de Matplotlib que contiene
                                      ambas subtramas (histograma y boxplot).
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # Histograma con una curva de densidad (KDE)
        sns.histplot(df[col], kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title(f'Distribución: {col}')
        
        # Boxplot para visualizar cuartiles y outliers
        sns.boxplot(x=df[col], ax=axes[1], color='lightgreen')
        axes[1].set_title(f'Boxplot: {col}')
        
        plt.tight_layout()
        return fig

    def create_categorical_count(self, df, col):
        """
        Crea un gráfico de barras para mostrar la frecuencia de cada categoría.

        Args:
            df (pd.DataFrame): El DataFrame que contiene los datos.
            col (str): El nombre de la columna categórica a visualizar.

        Returns:
            matplotlib.figure.Figure: La figura de Matplotlib con el gráfico de barras.
        """
        fig = plt.figure(figsize=(8, 4))
        # Ordena las barras por frecuencia descendente.
        sns.countplot(x=df[col], palette="viridis", order=df[col].value_counts().index)
        plt.title(f'Frecuencia: {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    def create_correlation_heatmap(self, corr_matrix):
        """
        Crea un mapa de calor para una matriz de correlación.

        Args:
            corr_matrix (pd.DataFrame): La matriz de correlación precalculada.

        Returns:
            matplotlib.figure.Figure or None: La figura de Matplotlib con el mapa de calor,
                                              o None si la matriz de entrada es None.
        """
        if corr_matrix is None: return None
        
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
        plt.title('Matriz de Correlación (Pearson)')
        plt.tight_layout()
        return fig