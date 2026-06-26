"""
migrate.py — One-time migration: copy computed tables from local SQLite to Supabase.
Run once: python migrate.py
"""

import sqlite3
import pandas as pd
from database import get_engine

TABLES = [
    # analysis result tables
    'rfm_segments',
    'clv_predictions',
    'clv_metrics',
    'feature_importance',
    'churn_scores',
    'campaign_roi',
    'campaign_uplift_detail',
    'cohort_results',
    'financial_scenarios',
    'cohort_retention',
    'weekly_trend',
    'monthly_segment_spend',
    'demographics_income',
    'demographics_age',
    'demographics_hh_comp',
    'clv_by_income',
    'brand_by_segment',
    'category_by_segment',
    'acquisition_trend',
    'acquisition_by_segment',
    'campaign_week_ranges',
    'discount_sensitivity',
    # views — read from SQLite, uploaded as tables to Supabase
    'category_spend',
]

def migrate():
    sqlite_con = sqlite3.connect('grocery_analysis.db')
    pg_engine  = get_engine()

    for table in TABLES:
        print(f"  Migrating {table} ...", end=' ', flush=True)
        df = pd.read_sql_query(f"SELECT * FROM {table}", sqlite_con)
        df.to_sql(table, pg_engine, if_exists='replace', index=False,
                  method='multi', chunksize=1000)
        print(f"{len(df):,} rows")

    # buyer_segments: derive from rfm_segments (campaign_engaged column)
    print("  Building buyer_segments from rfm_segments ...", end=' ', flush=True)
    rfm = pd.read_sql_query("SELECT household_key, total_spend, n_baskets, campaign_engaged FROM rfm_segments", sqlite_con)
    rfm['buyer_type'] = rfm['campaign_engaged'].map({1: 'Campaign', 0: 'Organic', True: 'Campaign', False: 'Organic'})
    buyer_seg = rfm[['household_key', 'total_spend', 'n_baskets', 'buyer_type']]
    buyer_seg.to_sql('buyer_segments', pg_engine, if_exists='replace', index=False, method='multi', chunksize=1000)
    print(f"{len(buyer_seg):,} rows")

    sqlite_con.close()
    print("\nMigration complete.")

if __name__ == '__main__':
    migrate()
