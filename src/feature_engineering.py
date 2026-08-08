import pandas as pd
import numpy as np
import os
from pathlib import Path

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
        self.customer_features = None
        self.product_features = None
        
        self.df_enriched = None

    def create_temporal_features(self):
            try:
                self.df['days'] = self.df['transaction_date'].dt.day
                self.df['months'] = self.df['transaction_date'].dt.month
                self.df['year'] = self.df['transaction_date'].dt.year
                self.df['quarter'] = self.df['transaction_date'].dt.quarter
                self.df['day_of_week'] = self.df['transaction_date'].dt.day_name()
                self.df['is_weekend'] = self.df['transaction_date'].dt.weekday >= 5
                self.df['is_weekday'] = self.df['transaction_date'].dt.weekday < 5
            except Exception as e:
                raise Exception(f"Erreur lors de la création des features temporelles : {e}")
    
            return self.df

   
        
    def create_financial_features(self):
        try:
            self.df['gross_revenue'] = self.df['quantity'] * self.df['unit_price']
            self.df['gross_revenue'] = self.df['gross_revenue'].round(2)
            self.df['discount_amount'] = self.df['gross_revenue'] * (self.df['discount_pct'] / 100)
            self.df['discount_amount'] = self.df['discount_amount'].round(2)
            self.df['net_revenue'] = self.df['gross_revenue'] - self.df['discount_amount']
            self.df['net_revenue'] = self.df['net_revenue'].round(2)
            self.df['discount'] = self.df['discount_pct'].apply(lambda x: 1 if x > 0 else 0)
            self.df['discount'] = self.df['discount'].astype(bool)
        except Exception as e:
            raise Exception(f"Erreur lors de la création des features financières : {e}")
        
        return self.df
    
    def create_customer_features(self):
        try:
            self.customer_features = self.df.groupby('customer_id').agg({
                'transaction_id': 'count',
                'sales_amount': ['sum', 'mean', 'max', 'min']
            }).round(2)
            self.customer_features.columns = [
                'customer_transaction_count',
                'customer_total_spend',
                'customer_avg_basket',
                'customer_max_basket',
                'customer_min_basket'
            ]
            self.customer_features = self.customer_features.reset_index()
            
            print(f"Features client créées : {len(self.customer_features)} clients")
            return self.customer_features
        except Exception as e:
            raise Exception(f"Erreur lors de la création des features client : {e}")
    
    def create_product_features(self):
        try:
            self.product_features = self.df.groupby('product_id').agg({
                'transaction_id': 'count',
                'quantity': ['sum', 'mean', 'max', 'min'],
                'sales_amount': ['sum', 'mean']
            }).round(2)

            self.product_features.columns = [
                'product_transaction_count',
                'product_total_quantity',
                'product_avg_quantity',
                'product_max_quantity',
                'product_min_quantity',
                'product_total_revenue',
                'product_avg_price'
            ]
            self.product_features = self.product_features.reset_index()
            
            print(f"Features produit créées : {len(self.product_features)} produits")
            return self.product_features
        except Exception as e:
            raise Exception(f"Erreur lors de la création des features produit : {e}")
    
    def merge_selected_features(self, merge_customer=True, merge_product=True):
        self.df_enriched = self.df.copy()
        try:
            if merge_customer and self.customer_features is not None:
                useful_cols = ['customer_id', 'customer_transaction_count', 'customer_total_spend']
                self.df_enriched = self.df_enriched.merge(
                    self.customer_features[useful_cols],
                    on='customer_id',
                    how='left'
                )
                print(f"Features client fusionnées")
            
            if merge_product and self.product_features is not None:
                useful_cols = ['product_id', 'product_total_revenue', 'product_transaction_count']
                self.df_enriched = self.df_enriched.merge(
                    self.product_features[useful_cols],
                    on='product_id',
                    how='left'
                )
                print(f"Features produit fusionnées")
        except Exception as e:
            raise Exception(f"Erreur lors de la fusion des features : {e}")
        
        return self.df_enriched
            

    def report_enriched_features(self, merge=True):
        if merge and self.df_enriched is not None:
            print(f"Shape du DataFrame enrichi : {self.df_enriched.shape}")
            report = {
                'original_shape': self.original_shape,
                'enriched_shape': self.df_enriched.shape,
                'customer_features': len(self.customer_features) if self.customer_features is not None else 0,
                'product_features': len(self.product_features) if self.product_features is not None else 0
            }
        else:
            report = {
                'original_shape': self.original_shape,
                'enriched_shape': None,
                'customer_features': len(self.customer_features) if self.customer_features is not None else 0,
                'product_features': len(self.product_features) if self.product_features is not None else 0
            }
        return report

    def create_all_features(self, merge=True):
        self.create_temporal_features()
        print("Features temporelles créées")
        self.create_financial_features()
        print("Features financières créées")

        self.create_customer_features()
        self.create_product_features()

        

        if merge:
            self.df_enriched = self.merge_selected_features()
            print(f"\nFeatures fusionnées - Shape: {self.df_enriched.shape}")
            report = self.report_enriched_features(merge=merge)
            return self.df_enriched, report
        else:
            print("\nFeatures créées sans fusion")
            report = self.report_enriched_features(merge=merge)
            return self.df, report


