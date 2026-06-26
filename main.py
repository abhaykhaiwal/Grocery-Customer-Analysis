"""
Customer Value & Campaign ROI Analysis
Dunnhumby — The Complete Journey (real US grocery loyalty card data)

Sections:
  1. Load & Clean Data
  2. Customer Feature Engineering
  3. RFM Segmentation
  4. Campaign A/B Test & ROI (pre/post within-household uplift)
  5. Customer Lifetime Value Prediction (XGBoost, Y1 features → Y2 spend)
  6. Churn Prediction (Logistic Regression, AUC on holdout)
  7. Cohort Retention Analysis
  8. Financial Targeting Model
  9. Save to Database
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, classification_report, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from database import (
    build_database, get_engine, query,
    save_rfm, save_clv, save_churn, save_campaign_roi,
    save_cohort, save_financial, save_clv_metrics, save_feature_importance,
)

pd.set_option('display.float_format', '{:.3f}'.format)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Assumptions (all explicitly labelled) ─────────────────────────────────────
# Source: DMA (Direct Marketing Association) 2023 industry benchmarks
MAILER_COST    = {'TypeA': 1.50, 'TypeB': 0.80, 'TypeC': 0.50}
GROCERY_MARGIN = 0.27   # US grocery avg gross margin (FMI 2023)

# ── Time boundaries ───────────────────────────────────────────────────────────
MAX_DAY       = 711
CAL_END_DAY   = 365     # Year 1 = calibration for CLV
CHURN_WINDOW  = 90      # days of inactivity = churned
CHURN_CUTOFF  = MAX_DAY - CHURN_WINDOW  # day 621

REF_DATE  = pd.Timestamp('2013-01-01')

RFM_ORDER  = ['Champions', 'Loyal', 'At Risk', 'Lost']
RFM_COLORS = {'Champions':'#2ecc71','Loyal':'#3498db','At Risk':'#f39c12','Lost':'#e74c3c'}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 1: Load & Clean Data")
print("="*65)

# Load raw CSVs into Supabase on first run; subsequent runs skip this step
build_database()

print("  Reading data from Supabase ...")
txn       = query("SELECT * FROM transactions")
camp_tbl  = query("SELECT * FROM campaign_assignments")
camp_desc = query("SELECT * FROM campaign_descriptions")
coupon_r  = query("SELECT * FROM coupon_redemptions")
demo      = query("SELECT * FROM households")
product   = query("SELECT * FROM products")

txn['date'] = REF_DATE + pd.to_timedelta(txn['day'] - 1, unit='D')

# Basket-level aggregation
basket = (
    txn.groupby(['household_key','basket_id','day','week_no','date'])
    .agg(spend=('sales_value','sum'), n_items=('quantity','sum'))
    .reset_index()
)
basket_y1 = basket[basket['day'] <= CAL_END_DAY].copy()
basket_y2 = basket[basket['day'] >  CAL_END_DAY].copy()

print(f"Transactions  : {len(txn):,}")
print(f"Baskets       : {len(basket):,}")
print(f"Households    : {basket['household_key'].nunique():,}")
print(f"Date range    : {basket['date'].min().date()} -> {basket['date'].max().date()}")
print(f"Total revenue : ${txn['sales_value'].sum():,.2f}")
print(f"Avg basket    : ${basket['spend'].mean():.2f}")
print(f"Year 1 baskets: {len(basket_y1):,}  |  Year 2: {len(basket_y2):,}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: CUSTOMER FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 2: Customer Feature Engineering")
print("="*65)

spend_stats = basket.groupby('household_key').agg(
    total_spend   = ('spend',      'sum'),
    n_baskets     = ('basket_id',  'count'),
    avg_basket    = ('spend',      'mean'),
    first_day     = ('day',        'min'),
    last_day      = ('day',        'max'),
).reset_index()
spend_stats['tenure_days']  = spend_stats['last_day'] - spend_stats['first_day']
spend_stats['recency_days'] = MAX_DAY - spend_stats['last_day']
spend_stats['weekly_freq']  = spend_stats['n_baskets'] / (MAX_DAY / 7)

# Year-split spend for CLV validation
y1_spend = basket_y1.groupby('household_key')['spend'].sum().rename('y1_spend')
y2_spend = basket_y2.groupby('household_key')['spend'].sum().rename('y2_spend')

# Campaign engagement
camp_engaged = camp_tbl.groupby('household_key')['campaign'].count().rename('n_campaigns')
coupon_redeemed = coupon_r.groupby('household_key')['day'].count().rename('n_redemptions')

# Product diversity (number of unique departments)
prod_dept = txn.merge(product[['product_id','department']], on='product_id', how='left')
dept_div  = prod_dept.groupby('household_key')['department'].nunique().rename('n_departments')

# Income encoding (midpoint of bracket, thousands of dollars)
income_map = {
    'Under 15K':12.5, '15-24K':19.5, '25-34K':29.5, '35-49K':42,
    '50-74K':62, '75-99K':87, '100-124K':112, '125-149K':137,
    '150-174K':162, '175-199K':187, '200-249K':224, '250K+':275,
}
age_map = {'19-24':21.5,'25-34':29.5,'35-44':39.5,'45-54':49.5,'55-64':59.5,'65+':70}
demo['income_val'] = demo['income_desc'].map(income_map)
demo['age_val']    = demo['age_desc'].map(age_map)

# Build customer master
customers = (
    spend_stats
    .join(y1_spend, on='household_key')
    .join(y2_spend, on='household_key')
    .join(camp_engaged, on='household_key')
    .join(coupon_redeemed, on='household_key')
    .join(dept_div, on='household_key')
    .merge(demo[['household_key','income_val','age_val']], on='household_key', how='left')
)
customers[['y1_spend','y2_spend','n_campaigns','n_redemptions']] = \
    customers[['y1_spend','y2_spend','n_campaigns','n_redemptions']].fillna(0)
customers['income_val']    = customers['income_val'].fillna(customers['income_val'].median())
customers['age_val']       = customers['age_val'].fillna(customers['age_val'].median())
customers['campaign_engaged'] = (customers['n_campaigns'] > 0).astype(int)

print(f"Customer features built for {len(customers):,} households")
print(f"  Avg total spend   : ${customers['total_spend'].mean():,.2f}")
print(f"  Avg baskets       : {customers['n_baskets'].mean():.1f}")
print(f"  In campaigns      : {customers['campaign_engaged'].sum():,} "
      f"({customers['campaign_engaged'].mean()*100:.1f}%)")
print(f"  Coupon redeemers  : {(customers['n_redemptions']>0).sum():,}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: RFM SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 3: RFM Segmentation")
print("="*65)

customers['r_score'] = pd.qcut(
    customers['recency_days'].rank(method='first'), q=5,
    labels=[5,4,3,2,1]
).astype(int)
customers['f_score'] = pd.qcut(
    customers['n_baskets'].rank(method='first'), q=5,
    labels=[1,2,3,4,5]
).astype(int)
customers['m_score'] = pd.qcut(
    customers['total_spend'].rank(method='first'), q=5,
    labels=[1,2,3,4,5]
).astype(int)
customers['rfm_score'] = customers['r_score'] + customers['f_score'] + customers['m_score']

def rfm_segment(s):
    if s >= 13: return 'Champions'
    if s >= 10: return 'Loyal'
    if s >= 7:  return 'At Risk'
    return 'Lost'

customers['rfm_segment'] = customers['rfm_score'].apply(rfm_segment)

rfm_counts = customers['rfm_segment'].value_counts()
rfm_pct    = rfm_counts / len(customers) * 100

rfm_profile = (
    customers.groupby('rfm_segment')
    .agg(n=('household_key','count'),
         avg_spend=('total_spend','mean'),
         avg_baskets=('n_baskets','mean'),
         avg_basket_size=('avg_basket','mean'),
         avg_recency=('recency_days','mean'),
         avg_rfm=('rfm_score','mean'))
    .reindex(RFM_ORDER).round(2)
)

print("\nRFM Segment Distribution:")
for seg in RFM_ORDER:
    n = rfm_counts.get(seg, 0)
    avg = rfm_profile.loc[seg, 'avg_spend'] if seg in rfm_profile.index else 0
    print(f"  {seg:<12}: {n:5,}  ({rfm_pct.get(seg,0):.1f}%)  avg spend ${avg:,.2f}")

print("\nRFM Segment Profiles:")
print(rfm_profile.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: CAMPAIGN A/B TEST & ROI
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 4: Campaign A/B Test & ROI Analysis")
print("="*65)
print("  Method: within-household pre/post comparison")
print("  Pre-period: 8 weeks before campaign start")
print("  Mailer costs: TypeA=$1.50, TypeB=$0.80, TypeC=$0.50 (DMA 2023)")

PRE_WEEKS = 8

uplift_rows = []
for _, cd in camp_desc.iterrows():
    cid        = cd['campaign']
    ctype      = cd['description']
    start      = cd['start_day']
    end        = cd['end_day']
    dur_weeks  = max((end - start) / 7, 1)
    pre_start  = max(start - PRE_WEEKS * 7, 1)
    pre_end    = start - 1

    treatment_hh = camp_tbl[camp_tbl['campaign'] == cid]['household_key'].values

    for hh in treatment_hh:
        hh_b = basket[basket['household_key'] == hh]

        pre_spend    = hh_b[(hh_b['day'] >= pre_start) & (hh_b['day'] <= pre_end)]['spend'].sum()
        during_spend = hh_b[(hh_b['day'] >= start)     & (hh_b['day'] <= end)    ]['spend'].sum()

        if pre_spend == 0:
            continue  # exclude HH with no pre-period activity

        pre_weekly    = pre_spend / PRE_WEEKS
        during_weekly = during_spend / dur_weeks
        uplift_weekly = during_weekly - pre_weekly

        uplift_rows.append({
            'campaign'          : cid,
            'campaign_type'     : ctype,
            'household_key'     : hh,
            'pre_weekly_spend'  : round(pre_weekly,    2),
            'during_weekly_spend': round(during_weekly, 2),
            'uplift_weekly'     : round(uplift_weekly,  2),
            'dur_weeks'         : dur_weeks,
            'total_uplift'      : round(uplift_weekly * dur_weeks, 2),
        })

uplift_df = pd.DataFrame(uplift_rows)

# Aggregate by campaign type
camp_roi_rows = []
for ctype, grp in uplift_df.groupby('campaign_type'):
    n_hh          = grp['household_key'].nunique()
    avg_uplift_wk = grp['uplift_weekly'].mean()
    avg_dur       = grp['dur_weeks'].mean()
    total_uplift  = grp['total_uplift'].sum()
    mailer_cost   = MAILER_COST[ctype]
    total_cost    = n_hh * mailer_cost
    gross_profit  = total_uplift * GROCERY_MARGIN
    net_roi       = gross_profit / total_cost if total_cost > 0 else 0
    rev_roi       = total_uplift / total_cost if total_cost > 0 else 0

    # Statistical significance (one-sample t-test: is uplift > 0?)
    t_stat, p_val = stats.ttest_1samp(grp['uplift_weekly'].dropna(), 0)

    camp_roi_rows.append({
        'campaign_type'       : ctype,
        'n_households'        : n_hh,
        'avg_pre_weekly_spend': round(grp['pre_weekly_spend'].mean(), 2),
        'avg_during_weekly_spend': round(grp['during_weekly_spend'].mean(), 2),
        'avg_uplift_weekly'   : round(avg_uplift_wk, 2),
        'avg_duration_weeks'  : round(avg_dur, 1),
        'total_incremental_rev': round(total_uplift, 2),
        'total_mailer_cost'   : round(total_cost, 2),
        'gross_profit'        : round(gross_profit, 2),
        'revenue_roi'         : round(rev_roi, 2),
        'net_roi'             : round(net_roi, 2),
        'p_value'             : round(p_val, 4),
        'significant'         : int(p_val < 0.05),
        'mailer_cost_per_hh'  : mailer_cost,
    })

camp_roi = pd.DataFrame(camp_roi_rows).sort_values('net_roi', ascending=False)

print("\nCampaign ROI by Type:")
print(f"{'Type':<8} {'N HH':>6} {'Pre $/wk':>10} {'During $/wk':>12} "
      f"{'Uplift $/wk':>12} {'Total Uplift':>13} {'Rev ROI':>8} {'Net ROI':>8} {'Sig':>5}")
print("-"*80)
for _, r in camp_roi.iterrows():
    print(f"{r['campaign_type']:<8} {r['n_households']:>6,} "
          f"{r['avg_pre_weekly_spend']:>10.2f} {r['avg_during_weekly_spend']:>12.2f} "
          f"{r['avg_uplift_weekly']:>+12.2f} ${r['total_incremental_rev']:>12,.0f} "
          f"{r['revenue_roi']:>8.2f}x {r['net_roi']:>8.2f}x "
          f"{'YES' if r['significant'] else 'NO':>5}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: CLV PREDICTION — XGBoost + 5-Fold Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 5: CLV Prediction — XGBoost + Optuna Tuning + 5-Fold CV")
print("="*65)
print("  Model    : XGBoost Regressor on Year-2 spend")
print("  Features : 17 Year-1 behavioural + demographic features")
print("  Target   : Year-2 spend (direct)")
print("  Tuning   : Optuna Bayesian search (50 trials, 5-fold CV)")
print("  CV       : 5-fold OOF on best params")

# ── Year-1 feature engineering ────────────────────────────────────────────────
y1_stats = basket_y1.groupby('household_key').agg(
    y1_total_spend = ('spend',     'sum'),
    y1_n_baskets   = ('basket_id', 'count'),
    y1_avg_basket  = ('spend',     'mean'),
    y1_std_basket  = ('spend',     'std'),
    y1_first_day   = ('day',       'min'),
    y1_last_day    = ('day',       'max'),
).reset_index()
y1_stats['y1_recency']    = CAL_END_DAY - y1_stats['y1_last_day']
y1_stats['y1_tenure']     = y1_stats['y1_last_day'] - y1_stats['y1_first_day']
y1_stats['y1_std_basket'] = y1_stats['y1_std_basket'].fillna(0)
y1_stats['y1_spend_cv']   = np.where(
    y1_stats['y1_avg_basket'] > 0,
    y1_stats['y1_std_basket'] / y1_stats['y1_avg_basket'], 0
)

HALF_DAY = CAL_END_DAY // 2
y1_h1 = basket_y1[basket_y1['day'] <= HALF_DAY].groupby('household_key')['spend'].sum().rename('y1_spend_h1')
y1_h2 = basket_y1[basket_y1['day'] >  HALF_DAY].groupby('household_key')['spend'].sum().rename('y1_spend_h2')
y1_stats = y1_stats.join(y1_h1, on='household_key').join(y1_h2, on='household_key')
y1_stats[['y1_spend_h1','y1_spend_h2']] = y1_stats[['y1_spend_h1','y1_spend_h2']].fillna(0)
y1_stats['y1_spend_trend'] = np.where(
    y1_stats['y1_spend_h1'] > 0,
    y1_stats['y1_spend_h2'] / y1_stats['y1_spend_h1'], 1.0
)

y1_last4wk = (
    basket_y1[basket_y1['day'] >= CAL_END_DAY - 28]
    .groupby('household_key')['spend'].sum()
    .rename('y1_last4wk_spend')
)

y1_disc = (
    txn[txn['day'] <= CAL_END_DAY]
    .groupby('household_key')['retail_disc']
    .sum().abs().rename('y1_total_disc')
)
y1_stats = y1_stats.join(y1_last4wk, on='household_key').join(y1_disc, on='household_key')
y1_stats['y1_last4wk_spend'] = y1_stats['y1_last4wk_spend'].fillna(0)
y1_stats['y1_total_disc']    = y1_stats['y1_total_disc'].fillna(0)
y1_stats['y1_disc_ratio']    = np.where(
    y1_stats['y1_total_spend'] > 0,
    y1_stats['y1_total_disc'] / y1_stats['y1_total_spend'], 0
)

y1_coupons = (
    coupon_r[coupon_r['day'] <= CAL_END_DAY]
    .groupby('household_key')['day'].count()
    .rename('y1_redemptions')
)

y2_target = basket_y2.groupby('household_key')['spend'].sum().rename('y2_spend')

model_df = (
    customers[['household_key','n_departments','campaign_engaged','income_val','age_val']]
    .merge(y1_stats, on='household_key', how='left')
    .join(y1_coupons, on='household_key')
    .join(y2_target,  on='household_key')
)
model_df['y1_redemptions'] = model_df['y1_redemptions'].fillna(0)
model_df['y2_spend']       = model_df['y2_spend'].fillna(0)
model_df = model_df.set_index('household_key')

FEATURE_COLS = [
    'y1_total_spend',  'y1_n_baskets',   'y1_avg_basket',   'y1_std_basket',
    'y1_spend_cv',     'y1_recency',      'y1_tenure',       'y1_spend_h1',
    'y1_spend_h2',     'y1_spend_trend',  'y1_last4wk_spend','y1_disc_ratio',
    'n_departments',   'campaign_engaged','y1_redemptions',  'income_val',
    'age_val',
]

X = model_df[FEATURE_COLS].fillna(0)
y = model_df['y2_spend']

# ── Optuna hyperparameter search ──────────────────────────────────────────────
N_FOLDS    = 5
N_TRIALS   = 50
kf         = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

def _objective(trial):
    params = dict(
        n_estimators      = 1000,
        learning_rate     = trial.suggest_float('learning_rate',    0.01,  0.2,  log=True),
        max_depth         = trial.suggest_int(  'max_depth',        3,     8),
        subsample         = trial.suggest_float('subsample',        0.5,   1.0),
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5,   1.0),
        min_child_weight  = trial.suggest_int(  'min_child_weight', 1,     20),
        reg_lambda        = trial.suggest_float('reg_lambda',       0.1,   10.0, log=True),
        reg_alpha         = trial.suggest_float('reg_alpha',        1e-3,  5.0,  log=True),
        gamma             = trial.suggest_float('gamma',            0.0,   5.0),
        objective         = 'reg:absoluteerror',
        n_jobs            = -1,
        random_state      = RANDOM_STATE,
        verbosity         = 0,
        early_stopping_rounds = 50,
    )
    fold_maes = []
    for train_idx, val_idx in kf.split(X):
        X_tr_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_tr_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        X_tr_es, X_val_es, y_tr_es, y_val_es = train_test_split(
            X_tr_cv, y_tr_cv, test_size=0.15, random_state=RANDOM_STATE
        )
        m = XGBRegressor(**params)
        m.fit(X_tr_es, y_tr_es, eval_set=[(X_val_es, y_val_es)], verbose=False)
        preds = m.predict(X_val_cv)
        fold_maes.append(float(np.abs(preds - y_val_cv.values).mean()))
    return float(np.mean(fold_maes))

print(f"\n  Running Optuna search ({N_TRIALS} trials × {N_FOLDS}-fold) ...")
study = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(_objective, n_trials=N_TRIALS, show_progress_bar=False)

best_params = study.best_params
print(f"  Best trial MAE : ${study.best_value:.2f}")
print(f"  Best params    : {best_params}")

XGB_PARAMS = dict(
    **best_params,
    n_estimators          = 1000,
    objective             = 'reg:absoluteerror',
    n_jobs                = -1,
    random_state          = RANDOM_STATE,
    verbosity             = 0,
    early_stopping_rounds = 50,
)

# ── 5-Fold Cross-Validation with best params ──────────────────────────────────

oof_preds                           = np.zeros(len(X))
cv_maes, cv_r2s, cv_top20s, cv_meds = [], [], [], []
best_iters                          = []

print(f"\n  Running {N_FOLDS}-fold cross-validation...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_tr_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
    y_tr_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

    # Early stopping on 15% of the fold's training slice
    X_tr_es, X_val_es, y_tr_es, y_val_es = train_test_split(
        X_tr_cv, y_tr_cv, test_size=0.15, random_state=RANDOM_STATE
    )

    xgb_fold = XGBRegressor(**XGB_PARAMS)
    xgb_fold.fit(X_tr_es, y_tr_es, eval_set=[(X_val_es, y_val_es)], verbose=False)
    best_iters.append(xgb_fold.best_iteration)

    fold_preds         = xgb_fold.predict(X_val_cv)
    oof_preds[val_idx] = fold_preds

    abs_errs   = np.abs(fold_preds - y_val_cv.values)
    fold_mae   = float(abs_errs.mean())
    fold_med   = float(np.median(abs_errs))
    fold_r2    = float(r2_score(y_val_cv, fold_preds))
    top_n      = max(1, int(len(y_val_cv) * 0.2))
    ta         = set(y_val_cv.nlargest(top_n).index)
    tp         = set(pd.Series(fold_preds, index=y_val_cv.index).nlargest(top_n).index)
    fold_top20 = len(ta & tp) / top_n

    cv_maes.append(fold_mae);  cv_meds.append(fold_med)
    cv_r2s.append(fold_r2);    cv_top20s.append(fold_top20)
    print(f"  Fold {fold}/{N_FOLDS}: best_iter={xgb_fold.best_iteration}  "
          f"MAE=${fold_mae:.2f}  R²={fold_r2:.4f}  Top-20%={fold_top20*100:.1f}%")

cv_mae_mean   = float(np.mean(cv_maes))
cv_mae_std    = float(np.std(cv_maes))
cv_median_mean= float(np.mean(cv_meds))
cv_r2_mean    = float(np.mean(cv_r2s))
cv_r2_std     = float(np.std(cv_r2s))
cv_top20_mean = float(np.mean(cv_top20s))
avg_best_iter = int(np.mean(best_iters))

# OOF global metrics
mae       = cv_mae_mean
median_ae = cv_median_mean
r2        = cv_r2_mean
rmse      = float(np.sqrt(np.mean((oof_preds - y.values)**2)))

top_n_all       = max(1, int(len(y) * 0.2))
top_actual_all  = set(y.nlargest(top_n_all).index)
top_pred_oof    = set(pd.Series(oof_preds, index=X.index).nlargest(top_n_all).index)
top20_precision = len(top_actual_all & top_pred_oof) / top_n_all

print(f"\n  CV Summary ({N_FOLDS}-fold):")
print(f"  MAE              : ${cv_mae_mean:.2f} ± ${cv_mae_std:.2f}")
print(f"  Median Abs Error : ${cv_median_mean:.2f}")
print(f"  RMSE             : ${rmse:.2f}")
print(f"  R²               : {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
print(f"  Top-20% Precision: {top20_precision*100:.1f}%")
print(f"  Avg best iter    : {avg_best_iter} trees")

# ── Final model — trained on full dataset with avg best n_estimators ──────────
print(f"\n  Training final model on all {len(X):,} households "
      f"(n_estimators={avg_best_iter})...")
xgb = XGBRegressor(
    n_estimators=avg_best_iter, learning_rate=0.03, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, reg_lambda=1.0,
    objective='reg:absoluteerror',
    n_jobs=-1, random_state=RANDOM_STATE,
    verbosity=0,
)
xgb.fit(X, y)
y_pred_all = xgb.predict(X)

# ── Feature importance ────────────────────────────────────────────────────────
feat_imp = pd.DataFrame({
    'feature'   : FEATURE_COLS,
    'importance': xgb.feature_importances_,
}).sort_values('importance', ascending=False).reset_index(drop=True)
print(f"\n  Feature Importances (top 10):")
print(feat_imp.head(10).to_string(index=False))

# ── Build results table ───────────────────────────────────────────────────────
# OOF predictions for honest scatter plot; final-model predictions for CLV scores
clv_results = pd.DataFrame({
    'household_key'  : model_df.index,
    'predicted_clv'  : oof_preds,
    'actual_y2_spend': y.values,
    'in_test_set'    : 1,
})
clv_results['abs_error'] = np.abs(oof_preds - y.values)

# Attach final-model CLV to customers master for downstream financial model
customers = customers.merge(
    pd.DataFrame({'household_key': model_df.index, 'predicted_clv': y_pred_all}),
    on='household_key', how='left'
)
customers['predicted_clv'] = customers['predicted_clv'].fillna(float(np.median(y_pred_all)))

# ── Naive baseline ────────────────────────────────────────────────────────────
naive_abs_err  = np.abs(X['y1_total_spend'].values - y.values)
naive_mae      = float(naive_abs_err.mean())
naive_median   = float(np.median(naive_abs_err))
rf_improvement = (1 - mae / naive_mae) * 100

print(f"\n  Naive baseline (predict Y2 = Y1 spend):")
print(f"  Naive MAE  : ${naive_mae:.2f}")
print(f"  XGBoost MAE: ${mae:.2f}")
print(f"  Improvement: {rf_improvement:.1f}% better than naive baseline")

clv_metrics = {
    'model'               : f'XGBoost + Optuna ({N_TRIALS}t) + {N_FOLDS}-fold CV (n={avg_best_iter})',
    'median_ae'           : round(cv_median_mean,  2),
    'mae'                 : round(cv_mae_mean,      2),
    'mae_std'             : round(cv_mae_std,       2),
    'rmse'                : round(rmse,             2),
    'r2'                  : round(cv_r2_mean,       4),
    'r2_std'              : round(cv_r2_std,        4),
    'top20_precision'     : round(top20_precision,  4),
    'n_folds'             : N_FOLDS,
    'median_predicted_clv': round(float(np.median(y_pred_all)), 2),
    'mean_predicted_clv'  : round(float(y_pred_all.mean()),     2),
    'n_train'             : len(X),
    'n_test'              : len(X),
    'naive_mae'           : round(naive_mae,        2),
    'naive_median_ae'     : round(naive_median,     2),
    'rf_improvement_pct'  : round(rf_improvement,   1),
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: CHURN PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 6: Churn Prediction — Logistic Regression")
print("="*65)
print(f"  Definition: no purchase in last {CHURN_WINDOW} days (day {CHURN_CUTOFF}–{MAX_DAY})")

# Churn label
customers['is_churned'] = (customers['last_day'] < CHURN_CUTOFF).astype(int)
churn_rate = customers['is_churned'].mean()
print(f"  Overall churn rate: {churn_rate*100:.1f}%")

# Features
# recency_days is excluded: it directly encodes the churn label (churn = recency > 90)
# Using only behavioral features that would be available before the churn window
feature_cols = ['n_baskets','total_spend','avg_basket',
                'tenure_days','campaign_engaged','n_departments',
                'income_val','age_val','weekly_freq']
X = customers[feature_cols].fillna(0)
y = customers['is_churned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
lr.fit(X_train_sc, y_train)

y_prob  = lr.predict_proba(X_test_sc)[:, 1]
y_pred  = lr.predict(X_test_sc)
auc     = roc_auc_score(y_test, y_prob)

print(f"\n  Test set AUC: {auc:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Active','Churned']))

# Predict on all customers
X_all_sc = scaler.transform(X.fillna(0))
customers['churn_probability'] = lr.predict_proba(X_all_sc)[:, 1]

# Revenue at risk
high_risk = customers[customers['churn_probability'] >= 0.70]
revenue_at_risk = high_risk['predicted_clv'].sum()
n_high_risk     = len(high_risk)

print(f"\n  High-risk customers (prob >= 70%): {n_high_risk:,}")
print(f"  Revenue at risk (predicted CLV)  : ${revenue_at_risk:,.2f}")

# Churn model coefficients
churn_coef = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': lr.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)
print(f"\n  Top churn predictors:")
print(churn_coef.head(6).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: COHORT RETENTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 7: Cohort Retention Analysis")
print("="*65)

# Cohort = first purchase quarter
customers['first_quarter'] = pd.to_datetime(
    REF_DATE + pd.to_timedelta(customers['first_day'] - 1, unit='D')
).dt.to_period('M').astype(str)

cohort = (
    customers.groupby('first_quarter')
    .agg(
        n_customers    = ('household_key',    'count'),
        avg_spend      = ('total_spend',      'mean'),
        avg_baskets    = ('n_baskets',        'mean'),
        avg_clv        = ('predicted_clv',    'mean'),
        churn_rate     = ('is_churned',       'mean'),
        pct_campaign   = ('campaign_engaged', 'mean'),
    )
    .reset_index()
    .sort_values('first_quarter')
)
cohort = cohort[cohort['n_customers'] >= 20]

print("\nCohort Analysis:")
print(cohort.round(2).to_string(index=False))

best_cohort  = cohort.loc[cohort['avg_spend'].idxmax(), 'first_quarter']
worst_cohort = cohort.loc[cohort['avg_spend'].idxmin(), 'first_quarter']
print(f"\nHighest-spend cohort : {best_cohort}  (${cohort['avg_spend'].max():,.2f} avg)")
print(f"Lowest-spend cohort  : {worst_cohort}  (${cohort['avg_spend'].min():,.2f} avg)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7b: COHORT RETENTION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 7b: Cohort Retention Matrix")
print("="*65)

basket['cal_month'] = pd.to_datetime(
    REF_DATE + pd.to_timedelta(basket['day'] - 1, unit='D')
).dt.to_period('M')

first_cal_month = basket.groupby('household_key')['cal_month'].min().rename('cohort_month')
basket_cm       = basket.merge(first_cal_month.reset_index(), on='household_key')
basket_cm['months_since'] = [
    (c.year - h.year) * 12 + (c.month - h.month)
    for c, h in zip(basket_cm['cal_month'], basket_cm['cohort_month'])
]
active_counts = (
    basket_cm.groupby(['cohort_month', 'months_since'])['household_key']
    .nunique().unstack(fill_value=0)
)
cohort_sizes  = first_cal_month.value_counts().sort_index()
valid_cohorts = cohort_sizes[cohort_sizes >= 20].index
active_counts = active_counts.loc[active_counts.index.isin(valid_cohorts)]
cohort_sizes  = cohort_sizes[cohort_sizes.index.isin(valid_cohorts)]
retention_pct = active_counts.div(cohort_sizes, axis=0).mul(100).round(1)

retention_long = (
    retention_pct.reset_index()
    .melt(id_vars='cohort_month', var_name='months_since', value_name='retention_pct')
)
retention_long['cohort_month'] = retention_long['cohort_month'].astype(str)
retention_long = retention_long.dropna(subset=['retention_pct'])

avg_m3  = float(retention_pct[3].mean())  if 3  in retention_pct.columns else 0.0
avg_m12 = float(retention_pct[12].mean()) if 12 in retention_pct.columns else 0.0
print(f"  Cohorts (>=20 HH)     : {len(valid_cohorts)}")
print(f"  Avg 3-month retention : {avg_m3:.1f}%")
print(f"  Avg 12-month retention: {avg_m12:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7c: SEASONALITY & TIME TRENDS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 7c: Seasonality & Time Trends")
print("="*65)

weekly_trend = (
    basket.groupby('week_no')
    .agg(
        total_spend  = ('spend',         'sum'),
        n_baskets    = ('basket_id',     'count'),
        n_households = ('household_key', 'nunique'),
        avg_basket   = ('spend',         'mean'),
    )
    .reset_index().round(2)
)

basket_seg = basket.merge(
    customers[['household_key', 'rfm_segment']], on='household_key', how='left'
)
basket_seg['cal_month_str'] = basket_seg['cal_month'].astype(str)
monthly_seg = (
    basket_seg.groupby(['cal_month_str', 'rfm_segment'])['spend']
    .sum().reset_index()
)
monthly_seg.columns = ['cal_month', 'rfm_segment', 'total_spend']
monthly_seg['total_spend'] = monthly_seg['total_spend'].round(2)

peak_week = int(weekly_trend.loc[weekly_trend['total_spend'].idxmax(), 'week_no'])
print(f"  Weekly rows        : {len(weekly_trend)}")
print(f"  Monthly x seg rows : {len(monthly_seg)}")
print(f"  Peak spend week    : week {peak_week}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7d: DEMOGRAPHICS ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 7d: Demographics Analysis")
print("="*65)

demo_ext = demo[[c for c in [
    'household_key', 'income_desc', 'age_desc', 'marital_status_code',
    'hh_comp_desc', 'homeowner_desc', 'household_size_desc',
] if c in demo.columns]].copy()
customers_demo = customers.merge(demo_ext, on='household_key', how='left')

rfm_income = (
    customers_demo.dropna(subset=['income_desc'])
    .groupby(['rfm_segment', 'income_desc'])['household_key']
    .count().reset_index(name='n_households')
)
rfm_age_df = (
    customers_demo.dropna(subset=['age_desc'])
    .groupby(['rfm_segment', 'age_desc'])['household_key']
    .count().reset_index(name='n_households')
)
rfm_hh_df = (
    customers_demo.dropna(subset=['hh_comp_desc'])
    .groupby(['rfm_segment', 'hh_comp_desc'])['household_key']
    .count().reset_index(name='n_households')
)
clv_by_income = (
    customers_demo.dropna(subset=['income_desc'])
    .groupby('income_desc')
    .agg(
        avg_clv   = ('predicted_clv', 'mean'),
        avg_spend = ('total_spend',   'mean'),
        n         = ('household_key', 'count'),
    )
    .reset_index().round(2)
)

print(f"  RFM x income rows  : {len(rfm_income)}")
print(f"  RFM x age rows     : {len(rfm_age_df)}")
print(f"  RFM x hh_comp rows : {len(rfm_hh_df)}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7e: PRODUCT ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 7e: Product Analytics")
print("="*65)

brand_txn = txn.merge(product[['product_id', 'brand']], on='product_id', how='left')
brand_seg = (
    brand_txn.merge(customers[['household_key', 'rfm_segment']], on='household_key', how='left')
    .groupby(['rfm_segment', 'brand'])['sales_value']
    .sum().reset_index()
)
seg_totals = brand_seg.groupby('rfm_segment')['sales_value'].sum().rename('seg_total')
brand_seg  = brand_seg.join(seg_totals, on='rfm_segment')
brand_seg['pct'] = (brand_seg['sales_value'] / brand_seg['seg_total'] * 100).round(2)

cat_seg = (
    prod_dept.merge(customers[['household_key', 'rfm_segment']], on='household_key', how='left')
    .groupby(['rfm_segment', 'department'])
    .agg(
        total_spend  = ('sales_value',   'sum'),
        n_households = ('household_key', 'nunique'),
    )
    .reset_index()
)
cat_seg['avg_spend_per_hh'] = (cat_seg['total_spend'] / cat_seg['n_households']).round(2)

print(f"  Brand x segment rows   : {len(brand_seg)}")
print(f"  Category x segment rows: {len(cat_seg)}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7f: CUSTOMER ACQUISITION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 7f: Customer Acquisition")
print("="*65)

customers['first_cal_month'] = pd.to_datetime(
    REF_DATE + pd.to_timedelta(customers['first_day'] - 1, unit='D')
).dt.to_period('M').astype(str)

acquisition = (
    customers.groupby('first_cal_month')
    .agg(
        n_new     = ('household_key', 'count'),
        avg_clv   = ('predicted_clv', 'mean'),
        avg_spend = ('total_spend',   'mean'),
    )
    .reset_index().round(2)
)
acq_by_seg = (
    customers.groupby(['first_cal_month', 'rfm_segment'])['household_key']
    .count().reset_index(name='n_new')
)

for _, r in acquisition.iterrows():
    print(f"  {r['first_cal_month']}: {int(r['n_new']):,} new HH  "
          f"avg CLV ${r['avg_clv']:,.2f}  avg spend ${r['avg_spend']:,.2f}")

# Campaign periods → week numbers (for trend annotation)
camp_weeks = camp_desc.copy()
camp_weeks['start_week'] = ((camp_weeks['start_day'] - 1) // 7) + 1
camp_weeks['end_week']   = ((camp_weeks['end_day']   - 1) // 7) + 1
camp_weeks = camp_weeks[['campaign', 'description', 'start_week', 'end_week',
                          'start_day', 'end_day']].copy()

# Discount sensitivity — Y1 discount depth × RFM segment × full-dataset spend
disc_y1 = (
    txn[txn['day'] <= CAL_END_DAY]
    .groupby('household_key')
    .agg(
        y1_disc_abs = ('retail_disc', lambda x: x.abs().sum()),
        y1_rev      = ('sales_value', 'sum'),
    )
    .reset_index()
)
disc_y1['disc_ratio'] = (disc_y1['y1_disc_abs'] / disc_y1['y1_rev']).fillna(0)
disc_y1['disc_bucket'] = pd.cut(
    disc_y1['disc_ratio'],
    bins=[0, 0.05, 0.10, 0.15, 0.20, 1.01],
    labels=['<5%', '5-10%', '10-15%', '15-20%', '20%+'],
    include_lowest=True,
)
disc_full = disc_y1.merge(
    customers[['household_key', 'rfm_segment', 'total_spend', 'predicted_clv']],
    on='household_key', how='left',
)
disc_sensitivity = (
    disc_full.dropna(subset=['disc_bucket', 'rfm_segment'])
    .groupby(['rfm_segment', 'disc_bucket'], observed=True)
    .agg(
        n_customers   = ('household_key', 'count'),
        avg_2yr_spend = ('total_spend',   'mean'),
        avg_clv       = ('predicted_clv', 'mean'),
        avg_disc_ratio= ('disc_ratio',    'mean'),
    )
    .reset_index().round(2)
)
disc_sensitivity['disc_bucket'] = disc_sensitivity['disc_bucket'].astype(str)
print(f"\n  Discount sensitivity rows: {len(disc_sensitivity)}")
print(disc_sensitivity.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: FINANCIAL TARGETING MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 8: Financial Targeting Model")
print("="*65)

# TypeA ROI from campaign analysis
typeA_row = camp_roi[camp_roi['campaign_type']=='TypeA'].iloc[0] if len(camp_roi[camp_roi['campaign_type']=='TypeA']) else None
typeB_row = camp_roi[camp_roi['campaign_type']=='TypeB'].iloc[0] if len(camp_roi[camp_roi['campaign_type']=='TypeB']) else None

typeA_rev_roi = typeA_row['revenue_roi'] if typeA_row is not None else 2.5
typeB_rev_roi = typeB_row['revenue_roi'] if typeB_row is not None else 1.8

n_champ   = int(rfm_counts.get('Champions', 0))
n_loyal   = int(rfm_counts.get('Loyal',     0))
n_atrisk  = int(rfm_counts.get('At Risk',   0))
n_lost    = int(rfm_counts.get('Lost',      0))
n_total   = len(customers)

# Scenario 1: Blanket TypeA to all customers
blanket_cost    = n_total * MAILER_COST['TypeA']
blanket_uplift  = (typeA_row['avg_uplift_weekly'] * typeA_row['avg_duration_weeks']
                   if typeA_row is not None else 0) * n_total
blanket_roi     = blanket_uplift / blanket_cost if blanket_cost > 0 else 0

# Scenario 2: Targeted — TypeA to At Risk only, TypeB to Champions+Loyal, nothing to Lost
targeted_cost   = (n_atrisk * MAILER_COST['TypeA'] +
                   (n_champ + n_loyal) * MAILER_COST['TypeB'])
targeted_uplift = (
    (typeA_row['avg_uplift_weekly'] * typeA_row['avg_duration_weeks']
     if typeA_row is not None else 0) * n_atrisk +
    (typeB_row['avg_uplift_weekly'] * typeB_row['avg_duration_weeks']
     if typeB_row is not None else 0) * (n_champ + n_loyal)
)
targeted_roi    = targeted_uplift / targeted_cost if targeted_cost > 0 else 0

# Scenario 3: CLV-gated — only send where predicted CLV > 2 × mailer cost
clv_gate = customers[customers['predicted_clv'] > 2 * MAILER_COST['TypeA']].copy()
gated_n    = len(clv_gate)
gated_cost = gated_n * MAILER_COST['TypeA']
gated_uplift = (typeA_row['avg_uplift_weekly'] * typeA_row['avg_duration_weeks']
                if typeA_row is not None else 0) * gated_n
gated_roi   = gated_uplift / gated_cost if gated_cost > 0 else 0

cost_saving  = blanket_cost - targeted_cost
rev_retained = targeted_uplift / blanket_uplift * 100 if blanket_uplift > 0 else 0

print(f"\nAssumptions:")
print(f"  Mailer costs  : TypeA=${MAILER_COST['TypeA']}, TypeB=${MAILER_COST['TypeB']}, "
      f"TypeC=${MAILER_COST['TypeC']} (DMA 2023)")
print(f"  Grocery margin: {GROCERY_MARGIN*100:.0f}% (FMI 2023)")

print(f"\nFinancial Scenario Comparison:")
print(f"{'Metric':<32} {'Blanket TypeA':>14} {'Targeted':>14} {'CLV-Gated':>14}")
print("-"*76)
print(f"{'Households targeted':<32} {n_total:>14,} {n_champ+n_loyal+n_atrisk:>14,} {gated_n:>14,}")
print(f"{'Total mailer cost ($)':<32} ${blanket_cost:>13,.0f} ${targeted_cost:>13,.0f} ${gated_cost:>13,.0f}")
print(f"{'Incremental revenue ($)':<32} ${blanket_uplift:>13,.0f} ${targeted_uplift:>13,.0f} ${gated_uplift:>13,.0f}")
print(f"{'Revenue ROI':<32} {blanket_roi:>14.1f}x {targeted_roi:>14.1f}x {gated_roi:>14.1f}x")
print(f"{'Cost saving vs blanket ($)':<32} {'--':>14} ${cost_saving:>13,.0f} ${blanket_cost-gated_cost:>13,.0f}")
print(f"\n  Targeted strategy retains {rev_retained:.0f}% of uplift at {cost_saving/blanket_cost*100:.0f}% lower cost.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: SAVE TO DATABASE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 9: Saving to Database")
print("="*65)

save_rfm(customers)
save_clv(clv_results[['household_key','predicted_clv','actual_y2_spend','abs_error','in_test_set']])
save_feature_importance(feat_imp)

churn_save = customers[['household_key','churn_probability','is_churned',
                         'rfm_segment','predicted_clv']].copy()
save_churn(churn_save)
save_campaign_roi(camp_roi)
save_cohort(cohort)

save_financial([
    {'scenario':'Blanket TypeA', 'n_households':n_total,
     'mailer_cost':round(blanket_cost), 'incremental_revenue':round(blanket_uplift),
     'revenue_roi':round(blanket_roi,2), 'cost_saving':0},
    {'scenario':'Targeted (RFM)', 'n_households':n_champ+n_loyal+n_atrisk,
     'mailer_cost':round(targeted_cost), 'incremental_revenue':round(targeted_uplift),
     'revenue_roi':round(targeted_roi,2), 'cost_saving':round(cost_saving)},
    {'scenario':'CLV-Gated', 'n_households':gated_n,
     'mailer_cost':round(gated_cost), 'incremental_revenue':round(gated_uplift),
     'revenue_roi':round(gated_roi,2), 'cost_saving':round(blanket_cost-gated_cost)},
])
save_clv_metrics(clv_metrics)

engine = get_engine()
uplift_df.to_sql('campaign_uplift_detail', engine, if_exists='replace', index=False, method='multi', chunksize=1000)
retention_long.to_sql('cohort_retention',         engine, if_exists='replace', index=False, method='multi', chunksize=1000)
weekly_trend.to_sql('weekly_trend',               engine, if_exists='replace', index=False, method='multi', chunksize=1000)
monthly_seg.to_sql('monthly_segment_spend',       engine, if_exists='replace', index=False, method='multi', chunksize=1000)
rfm_income.to_sql('demographics_income',          engine, if_exists='replace', index=False, method='multi', chunksize=1000)
rfm_age_df.to_sql('demographics_age',             engine, if_exists='replace', index=False, method='multi', chunksize=1000)
rfm_hh_df.to_sql('demographics_hh_comp',          engine, if_exists='replace', index=False, method='multi', chunksize=1000)
clv_by_income.to_sql('clv_by_income',             engine, if_exists='replace', index=False, method='multi', chunksize=1000)
brand_seg.to_sql('brand_by_segment',              engine, if_exists='replace', index=False, method='multi', chunksize=1000)
cat_seg.to_sql('category_by_segment',             engine, if_exists='replace', index=False, method='multi', chunksize=1000)
acquisition.to_sql('acquisition_trend',            engine, if_exists='replace', index=False, method='multi', chunksize=1000)
acq_by_seg.to_sql('acquisition_by_segment',       engine, if_exists='replace', index=False, method='multi', chunksize=1000)
camp_weeks.to_sql('campaign_week_ranges',          engine, if_exists='replace', index=False, method='multi', chunksize=1000)
disc_sensitivity.to_sql('discount_sensitivity',    engine, if_exists='replace', index=False, method='multi', chunksize=1000)
con2.commit(); con2.close()

print("  rfm_segments            saved")
print("  clv_predictions         saved")
print("  churn_scores            saved")
print("  campaign_roi            saved")
print("  campaign_uplift_detail  saved")
print("  cohort_results          saved")
print("  financial_scenarios     saved")
print("  clv_metrics             saved")
print("  cohort_retention        saved")
print("  weekly_trend            saved")
print("  monthly_segment_spend   saved")
print("  demographics_income     saved")
print("  demographics_age        saved")
print("  demographics_hh_comp    saved")
print("  clv_by_income           saved")
print("  brand_by_segment        saved")
print("  category_by_segment     saved")
print("  acquisition_trend       saved")
print("  acquisition_by_segment  saved")
print("  campaign_week_ranges    saved")
print("  discount_sensitivity    saved")
print(f"\n  Database : grocery_analysis.db")
print("  Dashboard: streamlit run dashboard.py")
