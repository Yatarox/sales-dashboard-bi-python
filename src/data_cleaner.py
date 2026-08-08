import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape

    def clean_date(self, date_column, date_format=None):
        if date_column not in self.df.columns:
            raise ValueError(f"La colonne '{date_column}' n'existe pas dans le DataFrame.")
        try:
            if date_format:
                self.df[date_column] = pd.to_datetime(self.df[date_column], format=date_format, errors='coerce')
            else:
                self.df[date_column] = pd.to_datetime(self.df[date_column], errors='coerce')
            self.df = self.df.dropna(subset=[date_column])
            
            return self.df
        except Exception as e:
            raise Exception(f"Erreur lors du nettoyage de la colonne '{date_column}': {e}")

    def normalize_string_column(self, column_name):
        if column_name not in self.df.columns:
            raise ValueError(f"La colonne '{column_name}' n'existe pas dans le DataFrame.")
        try:
            self.df[column_name] = self.df[column_name].astype(str).str.strip().str.lower()
            return self.df
        except Exception as e:
            raise Exception(f"Erreur lors de la normalisation de la colonne '{column_name}': {e}")

    def remove_duplicates(self):
        initial_count = self.df.shape[0]
        self.df = self.df.drop_duplicates()
        final_count = self.df.shape[0]
        removed_count = initial_count - final_count
        print(f"Nombre de doublons supprimés : {removed_count}")
        return self.df

    def handle_outliers(self, column, method='clip', threshold=1.5):
        if column not in self.df.columns:
            raise ValueError(f"La colonne '{column}' n'existe pas")
        
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        print(f"Outliers dans '{column}': {len(outliers)} lignes")
        print(f"   Limites: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        if method == 'clip':
            self.df[column] = self.df[column].clip(lower_bound, upper_bound)
            print(f"Outliers 'clippés'")
        elif method == 'remove':
            self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
            print(f"Outliers supprimés")
        
        return self.df

    def get_cleaning_report(self):
        report = {
            'original_shape': self.original_shape,
            'new_shape': self.df.shape,
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'dtypes': self.df.dtypes.to_dict()
        }
        return report

    def clean_all(self, list_date_columns=None, list_string_columns=None, list_outlier_columns=None, outlier_method='clip', outlier_threshold=1.5):
        if list_date_columns:
            for col in list_date_columns:
                self.clean_date(col)
        if list_string_columns:
            for col in list_string_columns:
                self.normalize_string_column(col)
        if list_outlier_columns:
            for col in list_outlier_columns:
                self.handle_outliers(col, method=outlier_method, threshold=outlier_threshold)
        self.remove_duplicates()
        get_report = self.get_cleaning_report()
        return self.df, get_report

if __name__ == "__main__":
    data = {
        'date': ['2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', None],
        'value': [10, 20, 30, 1000, 50],
        'category': ['A', 'B', 'A', 'B', 'C']
    }
    df = pd.DataFrame(data)
    
    cleaner = DataCleaner(df)
    cleaned_df, report = cleaner.clean_all(
        list_date_columns=['date'],
        list_string_columns=['category'],
        list_outlier_columns=['value'],
        outlier_method='clip',
        outlier_threshold=1.5
    )
    
    print("DataFrame nettoyé :")
    print(cleaned_df)
    print("\nRapport de nettoyage :")
    print(report)

