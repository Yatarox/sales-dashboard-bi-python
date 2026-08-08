import pandas as pd
import numpy as np
from pathlib import Path
import kagglehub
import os

class DataLoader:
    def __init__(self, dataset_path=None):
        if dataset_path:
            self.filepath = Path(dataset_path)
        else:
            download_path = kagglehub.dataset_download("noopurbhatt/retail-sales-dataset")
            self.filepath = Path(download_path)
        
        self.df = None
        
    def load_data(self):
        try:
            csv_files = list(self.filepath.glob("*.csv"))
            
            if not csv_files:
                if self.filepath.suffix == '.csv' and self.filepath.exists():
                    csv_files = [self.filepath]
                else:
                    raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {self.filepath}")
            
            self.df = pd.read_csv(csv_files[0])
            print(f"Données chargées avec succès : {self.df.shape}")
            return self.df
            
        except Exception as e:
            raise Exception(f"Erreur lors du chargement des données : {e}")
    
    def get_basic_info(self):
        if self.df is None:
            self.load_data()
            
        return {
            "shape": self.df.shape,
            "columns": self.df.columns.tolist(),
            "dtypes": self.df.dtypes.to_dict(),
            "describe": self.df.describe(include='all').to_dict()
        }
    
    def get_quality_report(self):
        if self.df is None:
            self.load_data()
            
        return {
            "missing_values": self.df.isnull().sum().to_dict(),
            "missing_percentage": (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            "duplicates": self.df.duplicated().sum(),
            "unique_values": self.df.nunique().to_dict(),
            "memory_usage": self.df.memory_usage(deep=True).sum() / 1024**2
        }
    
    def get_sample(self, n=5, method='head', random_state=42):
        if self.df is None:
            self.load_data()
            
        if method == 'head':
            return self.df.head(n)
        elif method == 'tail':
            return self.df.tail(n)
        elif method == 'sample':
            return self.df.sample(n, random_state=random_state)
        else:
            print(f"Méthode '{method}' inconnue. Utilisation de 'head'")
            return self.df.head(n)


if __name__ == "__main__":
    loader = DataLoader()
    
    print("=" * 50)
    print("📊 CHARGEMENT DES DONNÉES")
    print("=" * 50)
    df = loader.load_data()
    
    print("\n📋 PREMIÈRES LIGNES :")
    print(loader.get_sample(5, 'head'))
    
    print("\nℹ️ INFORMATIONS DE BASE :")
    info = loader.get_basic_info()
    print(f"Shape : {info['shape']}")
    print(f"Colonnes : {info['columns']}")
    
    print("\n📊 RAPPORT QUALITÉ :")
    quality = loader.get_quality_report()
    print(f"Valeurs manquantes : {quality['missing_values']}")
    print(f"Doublons : {quality['duplicates']}")
    
    print("\n🎲 ÉCHANTILLON ALÉATOIRE :")
    print(loader.get_sample(3, 'sample'))