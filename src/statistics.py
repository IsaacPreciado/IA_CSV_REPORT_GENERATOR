import numpy as np

class StatsAnalyzer:
    def __init__(self, df, numerical_cols, categorical_cols):
        self.df = df
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols

    def get_missing_percentage(self):
        """Calcula porcentaje de nulos."""
        missing = self.df.isnull().mean() * 100
        return missing[missing > 0].sort_values(ascending=False)

    def get_categorical_modes(self):
        """Obtiene la moda de categóricas (ignorando textos libres)."""
        modes = {}
        for col in self.categorical_cols:
            if self.df[col].nunique() > 50: continue # Ignorar nombres/textos largos
            modes[col] = self.df[col].mode()[0]
        return modes

    def calculate_correlations(self):
        """Calcula Pearson y devuelve top 5 con signo real."""
        if len(self.numerical_cols) < 2:
            return {}, None

        # Matriz Pearson
        corr_matrix = self.df[self.numerical_cols].corr(method='pearson')
        
        #Eliminar diplucados
        mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        corrs = corr_matrix.where(mask).stack()
        
        # Ordenar por magnitud absoluta, pero devolver valor real
        top_corr = corrs.iloc[np.argsort(-corrs.abs())].head(5)
        
        # Retornamos diccionario para IA y matriz para gráfica
        top_dict = {f"{i[0]} vs {i[1]}": round(v, 4) for i, v in top_corr.items()}
        return top_dict, corr_matrix

    def count_outliers_iqr(self):
        """Cuenta outliers totales usando IQR."""
        total_outliers = 0
        for col in self.numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            currentOutliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
            total_outliers += currentOutliers
        return total_outliers