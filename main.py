import pandas as pd
import os
from pathlib import Path

from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.feature_engineering import FeatureEngineer


def run_pipeline():
    print("=" * 60)
    print("DEMARRAGE DU PIPELINE DATA SCIENCE")
    print("=" * 60)
    
    # 1. Chargement des données
    print("\nETAPE 1 : Chargement des donnees")
    loader = DataLoader()
    df = loader.load_data()
    print(f"   Donnees chargees : {df.shape}")
    
    # 2. Nettoyage des données
    print("\nETAPE 2 : Nettoyage des donnees")
    cleaner = DataCleaner(df)
    df_cleaned, report = cleaner.clean_all(
        list_date_columns=['transaction_date'],
        list_string_columns=['customer_id', 'product_id'],
        list_outlier_columns=['quantity', 'unit_price'],
        outlier_method='clip',
        outlier_threshold=1.5
    )
    print(f"   Donnees nettoyees : {df_cleaned.shape}")
    
    # 3. Feature Engineering
    print("\nETAPE 3 : Creation des features")
    engineer = FeatureEngineer(df_cleaned)
    enriched_df, feature_report = engineer.create_all_features(merge=True)
    print(f"   Features creees : {enriched_df.shape}")
    
    # 4. Sauvegarde
    print("\nETAPE 4 : Sauvegarde des donnees")
    output_path = "data/processed/enriched_data.csv"
    os.makedirs("data/processed", exist_ok=True)
    enriched_df.to_csv(output_path, index=False)
    print(f"   Donnees sauvegardees : {output_path}")
    
    # 5. Rapport final
    print("\n" + "=" * 60)
    print("RAPPORT FINAL")
    print("=" * 60)
    print(f"\nStatistiques :")
    print(f"   - Transactions : {enriched_df.shape[0]:,}")
    print(f"   - Colonnes : {enriched_df.shape[1]}")
    print(f"   - Clients uniques : {feature_report['customer_features']:,}")
    print(f"   - Produits uniques : {feature_report['product_features']}")
    
    print("\nColonnes ajoutees :")
    original_cols = loader.get_basic_info()['columns']
    new_cols = [col for col in enriched_df.columns if col not in original_cols]
    for col in new_cols:
        print(f"   - {col}")
    
    print("\n" + "=" * 60)
    print("PIPELINE TERMINE AVEC SUCCES")
    print("=" * 60)
    
    return enriched_df, report, feature_report


if __name__ == "__main__":
    df, clean_report, feature_report = run_pipeline()
    
    print("\nApercu des donnees :")
    print(df.head())