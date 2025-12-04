import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self):
        sns.set_theme(style="whitegrid")
        # No fijamos el tamaño global aquí para que sea dinámico, 
        # pero usamos defaults en cada método.

    def create_missing_heatmap(self, df):
        fig = plt.figure(figsize=(10, 8.5))
        sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
        plt.title('Mapa de Valores Faltantes')
        plt.tight_layout()
        return fig

    def create_numerical_distributions(self, df, col):
        """Genera figura para UNA columna numérica."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        sns.histplot(df[col], kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title(f'Distribución: {col}')
        
        sns.boxplot(x=df[col], ax=axes[1], color='lightgreen')
        axes[1].set_title(f'Boxplot: {col}')
        
        plt.tight_layout()
        return fig

    def create_categorical_count(self, df, col):
        """Genera figura para UNA columna categórica."""
        fig = plt.figure(figsize=(8, 4))
        sns.countplot(x=df[col], palette="viridis", order=df[col].value_counts().index)
        plt.title(f'Frecuencia: {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    def create_correlation_heatmap(self, corr_matrix):
        if corr_matrix is None: return None
        
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
        plt.title('Matriz de Correlación (Pearson)')
        plt.tight_layout()
        return fig