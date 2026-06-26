"""
database.py — Supabase (PostgreSQL) layer for Dunnhumby Complete Journey Analysis
"""

import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(Path(__file__).parent / '.env', override=True)

import psycopg2

def _db_config() -> dict:
    """Return connection params — Streamlit secrets take priority over .env."""
    try:
        import streamlit as st
        s = st.secrets
        return dict(host=s['DB_HOST'], port=int(s['DB_PORT']),
                    dbname=s['DB_NAME'], user=s['DB_USER'],
                    password=s['DB_PASSWORD'])
    except Exception:
        return dict(host=os.environ['DB_HOST'], port=int(os.environ['DB_PORT']),
                    dbname=os.environ['DB_NAME'], user=os.environ['DB_USER'],
                    password=os.environ['DB_PASSWORD'])

def get_engine():
    def _connect():
        return psycopg2.connect(**_db_config(), sslmode='require')
    return create_engine('postgresql+psycopg2://', creator=_connect)


def query(sql: str) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql_query(text(sql), conn)


def db_exists() -> bool:
    try:
        with get_engine().connect() as conn:
            result = conn.execute(text(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = 'rfm_segments'"
            ))
            return result.fetchone() is not None
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Build raw tables from CSVs (run once — skips if tables already exist)
# ─────────────────────────────────────────────────────────────────────────────

def build_database(force_rebuild: bool = False) -> None:
    engine = get_engine()

    if not force_rebuild:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = 'transactions'"
            ))
            if result.fetchone() is not None:
                print("  Raw tables already in Supabase — skipping CSV load.")
                return

    print("\nLoading CSV files into Supabase ...")

    txn       = pd.read_csv('transaction_data.csv')
    camp_tbl  = pd.read_csv('campaign_table.csv')
    camp_desc = pd.read_csv('campaign_desc.csv')
    coupon    = pd.read_csv('coupon.csv')
    coupon_r  = pd.read_csv('coupon_redempt.csv')
    demo      = pd.read_csv('hh_demographic.csv')
    product   = pd.read_csv('product.csv')

    for df in [txn, camp_tbl, camp_desc, coupon, coupon_r, demo, product]:
        df.columns = df.columns.str.lower()

    print("  Uploading transactions ...")
    txn[[
        'household_key', 'basket_id', 'day', 'week_no', 'product_id',
        'quantity', 'sales_value', 'retail_disc', 'coupon_disc',
        'coupon_match_disc', 'store_id',
    ]].to_sql('transactions', engine, if_exists='replace', index=False,
              method='multi', chunksize=5000)

    print("  Uploading reference tables ...")
    demo.to_sql('households',             engine, if_exists='replace', index=False, method='multi', chunksize=5000)
    camp_tbl.to_sql('campaign_assignments',  engine, if_exists='replace', index=False, method='multi', chunksize=5000)
    camp_desc.to_sql('campaign_descriptions',engine, if_exists='replace', index=False, method='multi', chunksize=5000)
    coupon.to_sql('coupons',              engine, if_exists='replace', index=False, method='multi', chunksize=5000)
    coupon_r.to_sql('coupon_redemptions', engine, if_exists='replace', index=False, method='multi', chunksize=5000)
    product[['product_id', 'department', 'brand', 'commodity_desc', 'sub_commodity_desc']].to_sql(
        'products', engine, if_exists='replace', index=False, method='multi', chunksize=5000
    )

    print("  Creating indexes and views ...")
    with engine.connect() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_txn_hh   ON transactions(household_key)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_txn_day  ON transactions(day)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_txn_prod ON transactions(product_id)"))

        conn.execute(text("""
            CREATE OR REPLACE VIEW customer_summary AS
            SELECT
                household_key,
                COUNT(DISTINCT basket_id)                               AS n_baskets,
                ROUND(SUM(basket_spend)::numeric, 2)                    AS total_spend,
                ROUND(AVG(basket_spend)::numeric, 2)                    AS avg_basket,
                MIN(day)                                                AS first_day,
                MAX(day)                                                AS last_day,
                MAX(day) - MIN(day)                                     AS tenure_days,
                711 - MAX(day)                                          AS recency_days
            FROM (
                SELECT household_key, basket_id, day,
                       SUM(sales_value) AS basket_spend
                FROM transactions
                GROUP BY household_key, basket_id, day
            ) s
            GROUP BY household_key
        """))

        conn.execute(text("""
            CREATE OR REPLACE VIEW campaign_summary AS
            SELECT
                ca.description,
                COUNT(DISTINCT ca.household_key) AS n_households,
                COUNT(DISTINCT ca.campaign)      AS n_campaigns
            FROM campaign_assignments ca
            GROUP BY ca.description
        """))

        conn.execute(text("""
            CREATE OR REPLACE VIEW weekly_spend AS
            SELECT
                household_key,
                week_no,
                ROUND(SUM(sales_value)::numeric, 2) AS weekly_spend,
                COUNT(DISTINCT basket_id)           AS n_baskets
            FROM transactions
            GROUP BY household_key, week_no
        """))

        conn.execute(text("""
            CREATE OR REPLACE VIEW category_spend AS
            SELECT
                p.department,
                COUNT(DISTINCT t.basket_id)         AS n_baskets,
                ROUND(SUM(t.sales_value)::numeric, 2)  AS total_spend,
                ROUND(AVG(t.sales_value)::numeric, 2)  AS avg_item_spend,
                COUNT(DISTINCT t.household_key)     AS n_households
            FROM transactions t
            JOIN products p ON t.product_id = p.product_id
            GROUP BY p.department
        """))

        conn.execute(text("""
            CREATE OR REPLACE VIEW buyer_segments AS
            SELECT
                cs.household_key,
                cs.total_spend,
                cs.n_baskets,
                CASE WHEN ca.household_key IS NOT NULL
                     THEN 'Campaign' ELSE 'Organic' END AS buyer_type
            FROM customer_summary cs
            LEFT JOIN (SELECT DISTINCT household_key FROM campaign_assignments) ca
                ON cs.household_key = ca.household_key
        """))

        conn.commit()

    print("  Supabase database ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Save analysis results
# ─────────────────────────────────────────────────────────────────────────────

def _save(df: pd.DataFrame, table: str) -> None:
    df.to_sql(table, get_engine(), if_exists='replace', index=False,
              method='multi', chunksize=1000)


def save_rfm(customers_df: pd.DataFrame) -> None:
    cols = ['household_key', 'rfm_segment', 'rfm_score', 'r_score', 'f_score', 'm_score',
            'recency_days', 'n_baskets', 'total_spend', 'avg_basket', 'tenure_days',
            'campaign_engaged', 'predicted_clv', 'churn_probability', 'is_churned']
    _save(customers_df[[c for c in cols if c in customers_df.columns]], 'rfm_segments')


def save_clv(clv_df: pd.DataFrame) -> None:
    _save(clv_df, 'clv_predictions')


def save_churn(churn_df: pd.DataFrame) -> None:
    _save(churn_df, 'churn_scores')


def save_campaign_roi(roi_df: pd.DataFrame) -> None:
    _save(roi_df, 'campaign_roi')


def save_cohort(cohort_df: pd.DataFrame) -> None:
    _save(cohort_df, 'cohort_results')


def save_financial(financial_list: list) -> None:
    _save(pd.DataFrame(financial_list), 'financial_scenarios')


def save_clv_metrics(metrics: dict) -> None:
    _save(pd.DataFrame([metrics]), 'clv_metrics')


def save_feature_importance(fi_df: pd.DataFrame) -> None:
    _save(fi_df, 'feature_importance')
