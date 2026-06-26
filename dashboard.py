"""
dashboard.py — Streamlit Dashboard
Customer Value & Campaign ROI Analysis
Dunnhumby — The Complete Journey

Run: streamlit run dashboard.py
Requires: python main.py  (to populate grocery_analysis.db)
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from database import query, db_exists

st.set_page_config(
    page_title="Grocery Customer Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

RFM_COLORS = {
    'Champions':'#2ecc71','Loyal':'#3498db','At Risk':'#f39c12','Lost':'#e74c3c',
}
RFM_ORDER        = ['Champions','Loyal','At Risk','Lost']
CAMP_COLORS      = {'TypeA':'#e74c3c','TypeB':'#f39c12','TypeC':'#3498db'}
SCENARIO_COLORS  = {'Blanket TypeA':'#e74c3c','Targeted (RFM)':'#f39c12','CLV-Gated':'#27ae60'}

if not db_exists():
    st.error("Supabase data not found. Please run `python main.py` first to load data.")
    st.code("python main.py")
    st.stop()

# ── Cached loaders ───────────────────────────────────────────────────────────
@st.cache_data
def load_rfm():
    return query("SELECT * FROM rfm_segments")

@st.cache_data
def load_clv():
    return query("SELECT * FROM clv_predictions")

@st.cache_data
def load_clv_metrics():
    return query("SELECT * FROM clv_metrics").iloc[0].to_dict()

@st.cache_data
def load_feature_importance():
    return query("SELECT * FROM feature_importance ORDER BY importance DESC")

@st.cache_data
def load_churn():
    return query("SELECT * FROM churn_scores")

@st.cache_data
def load_campaign_roi():
    return query("SELECT * FROM campaign_roi ORDER BY revenue_roi DESC")

@st.cache_data
def load_uplift_detail():
    return query("SELECT * FROM campaign_uplift_detail")

@st.cache_data
def load_cohort():
    return query("SELECT * FROM cohort_results ORDER BY first_quarter")

@st.cache_data
def load_financial():
    return query("SELECT * FROM financial_scenarios")

@st.cache_data
def load_category():
    return query("SELECT * FROM category_spend ORDER BY total_spend DESC LIMIT 10")

@st.cache_data
def load_retention():
    return query("SELECT * FROM cohort_retention ORDER BY cohort_month, months_since")

@st.cache_data
def load_weekly_trend():
    return query("SELECT * FROM weekly_trend ORDER BY week_no")

@st.cache_data
def load_monthly_seg():
    return query("SELECT * FROM monthly_segment_spend ORDER BY cal_month")

@st.cache_data
def load_rfm_income():
    return query("SELECT * FROM demographics_income")

@st.cache_data
def load_rfm_age():
    return query("SELECT * FROM demographics_age")

@st.cache_data
def load_rfm_hh():
    return query("SELECT * FROM demographics_hh_comp")

@st.cache_data
def load_clv_income():
    return query("SELECT * FROM clv_by_income")

@st.cache_data
def load_brand_seg():
    return query("SELECT * FROM brand_by_segment")

@st.cache_data
def load_cat_seg():
    return query("SELECT * FROM category_by_segment ORDER BY total_spend DESC")

@st.cache_data
def load_acquisition():
    return query("SELECT * FROM acquisition_trend ORDER BY first_cal_month")

@st.cache_data
def load_acq_by_seg():
    return query("SELECT * FROM acquisition_by_segment ORDER BY first_cal_month")

@st.cache_data
def load_camp_weeks():
    return query("SELECT * FROM campaign_week_ranges")

@st.cache_data
def load_disc_sensitivity():
    return query("SELECT * FROM discount_sensitivity")

rfm      = load_rfm()
clv      = load_clv()
clv_m    = load_clv_metrics()
feat_imp = load_feature_importance()
churn    = load_churn()
camp_roi = load_campaign_roi()
uplift   = load_uplift_detail()
cohort   = load_cohort()
fin      = load_financial()
cat      = load_category()
retention  = load_retention()
wkly       = load_weekly_trend()
mthly_seg  = load_monthly_seg()
rfm_inc    = load_rfm_income()
rfm_age    = load_rfm_age()
rfm_hh_df  = load_rfm_hh()
clv_inc    = load_clv_income()
brand_df   = load_brand_seg()
cat_df     = load_cat_seg()
acq        = load_acquisition()
acq_seg    = load_acq_by_seg()
camp_wks   = load_camp_weeks()
disc_sens  = load_disc_sensitivity()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:0'>🛒 Customer Value & Campaign ROI Analysis</h1>",
    unsafe_allow_html=True,
)
st.caption(
    f"Dunnhumby — The Complete Journey · {rfm['household_key'].nunique():,} households · "
    "Real US grocery loyalty card data · 2-year observation window"
)
st.divider()

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📋 Executive Summary",
    "👥 RFM Segments",
    "📣 Campaign ROI",
    "💰 CLV Predictions",
    "⚠️ Churn Risk",
    "📅 Trends",
    "🧑‍🤝‍🧑 Demographics",
    "🛒 Products",
    "🗄️ SQL Explorer",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Key findings at a glance")

    blanket = fin[fin['scenario']=='Blanket TypeA'].iloc[0]
    targeted = fin[fin['scenario']=='Targeted (RFM)'].iloc[0]

    n_high_risk     = int((churn['churn_probability'] >= 0.70).sum())
    revenue_at_risk = churn[churn['churn_probability'] >= 0.70]['predicted_clv'].sum()
    churn_rate      = rfm['is_churned'].mean() * 100 if 'is_churned' in rfm.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Households",     f"{rfm['household_key'].nunique():,}",
              help="Households with at least 1 purchase")
    c2.metric("Avg Predicted CLV",    f"${clv_m.get('mean_predicted_clv', 0):,.2f}",
              help="52-week predicted revenue per household (BG/NBD model)")
    c3.metric("Revenue at Risk",      f"${revenue_at_risk:,.0f}",
              help=f"{n_high_risk:,} households with churn probability ≥ 70%")
    c4.metric("CLV vs Naive Baseline",  f"{clv_m.get('rf_improvement_pct', 0):.1f}% better",
              help="RF MAE vs naive 'predict Y2 = Y1 spend' baseline on held-out test set")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        roi_impr = (targeted['revenue_roi'] / blanket['revenue_roi']
                    if blanket['revenue_roi'] > 0 else 0)
        cost_sav = blanket['mailer_cost'] - targeted['mailer_cost']
        sav_pct  = cost_sav / blanket['mailer_cost'] * 100 if blanket['mailer_cost'] > 0 else 0

        with st.container(border=True):
            rfm_vc = rfm['rfm_segment'].value_counts()
            champ_pct = rfm_vc.get('Champions',0) / len(rfm) * 100
            lost_pct  = rfm_vc.get('Lost',0)      / len(rfm) * 100
            st.markdown(
                f"**Finding 1 — Top 20% of customers drive most revenue**  \n"
                f"Champions ({champ_pct:.0f}%) and Loyals generate the highest lifetime value.  \n"
                f"Lost segment ({lost_pct:.0f}%) have low recency — re-engagement cost exceeds CLV.  \n"
                f"*Action: protect Champions, activate At Risk with targeted campaigns.*"
            )
        with st.container(border=True):
            best  = camp_roi.iloc[0]
            st.markdown(
                f"**Finding 2 — Campaign ROI varies by type**  \n"
                f"{best['campaign_type']} delivers highest revenue ROI ({best['revenue_roi']:.2f}x).  \n"
                f"Uplift measured via pre/post within-household comparison (n={best['n_households']:,}).  \n"
                f"*Mailer costs from DMA 2023 benchmarks (stated assumption).*"
            )
        with st.container(border=True):
            precision = clv_m.get('top20_precision', 0) * 100
            st.markdown(
                f"**Finding 3 — CLV model validated on held-out year**  \n"
                f"BG/NBD + Gamma-Gamma trained on Year 1, validated on Year 2.  \n"
                f"MAE=${clv_m.get('mae',0):.2f} · Top-20% precision={precision:.0f}%.  \n"
                f"*{precision:.0f}% of predicted top spenders were actual top spenders.*"
            )
        with st.container(border=True):
            st.markdown(
                f"**Finding 4 — Targeted strategy: {sav_pct:.0f}% lower cost, similar uplift**  \n"
                f"Blanket TypeA ROI: {blanket['revenue_roi']:.1f}x → "
                f"Targeted ROI: {targeted['revenue_roi']:.1f}x.  \n"
                f"Cost saving: ${cost_sav:,.0f} by excluding low-CLV households.  \n"
                f"*Revenue at risk from churning customers: ${revenue_at_risk:,.0f}.*"
            )

    with col_r:
        # RFM donut
        rfm_vc_df = rfm['rfm_segment'].value_counts().reset_index()
        rfm_vc_df.columns = ['segment','count']
        rfm_vc_df = rfm_vc_df[rfm_vc_df['segment'].isin(RFM_ORDER)]
        fig = px.pie(rfm_vc_df, names='segment', values='count',
                     color='segment', color_discrete_map=RFM_COLORS,
                     hole=0.4, title='Customer Segments')
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(showlegend=False, height=280)
        st.plotly_chart(fig, use_container_width=True)

        # Financial ROI bars
        fig2 = go.Figure()
        fig2.add_bar(
            x=fin['scenario'], y=fin['revenue_roi'],
            marker_color=[SCENARIO_COLORS.get(s,'#888') for s in fin['scenario']],
            text=[f"{v:.1f}x" for v in fin['revenue_roi']],
            textposition='outside',
        )
        fig2.add_hline(y=1, line_dash='dash', line_color='black',
                       annotation_text='Break-even')
        fig2.update_layout(title='Revenue ROI by Targeting Strategy',
                           yaxis_title='Revenue ROI (Uplift / Cost)',
                           showlegend=False, height=280)
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RFM SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("RFM Segmentation — Customer Value Tiers")
    st.markdown(
        "Each household scored on **Recency** (days since last shop), "
        "**Frequency** (number of baskets), and **Monetary** (total spend). "
        "Quintile scores 1–5 on each → composite segment."
    )

    rfm_vc = rfm['rfm_segment'].value_counts()
    c1,c2,c3,c4 = st.columns(4)
    for col, seg in zip([c1,c2,c3,c4], RFM_ORDER):
        n = rfm_vc.get(seg, 0)
        col.metric(seg, f"{n:,}", f"{n/len(rfm)*100:.1f}%")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        rfm_vc_df = rfm_vc.reset_index()
        rfm_vc_df.columns = ['segment','count']
        rfm_vc_df = rfm_vc_df[rfm_vc_df['segment'].isin(RFM_ORDER)]
        fig = px.pie(rfm_vc_df, names='segment', values='count',
                     color='segment', color_discrete_map=RFM_COLORS,
                     hole=0.4, title='Segment Sizes')
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(showlegend=False, height=370)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        seg_spend = rfm.groupby('rfm_segment')['total_spend'].mean().reindex(RFM_ORDER)
        fig = px.bar(seg_spend.reset_index(), x='rfm_segment', y='total_spend',
                     color='rfm_segment', color_discrete_map=RFM_COLORS,
                     title='Avg 2-Year Spend per Segment ($)',
                     labels={'rfm_segment':'Segment','total_spend':'Avg Spend ($)'},
                     category_orders={'rfm_segment': RFM_ORDER},
                     text_auto='$.0f')
        fig.update_layout(showlegend=False, height=370)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.box(rfm[rfm['rfm_segment'].isin(RFM_ORDER)],
                 x='rfm_segment', y='rfm_score',
                 color='rfm_segment', color_discrete_map=RFM_COLORS,
                 category_orders={'rfm_segment': RFM_ORDER},
                 title='RFM Score Distribution by Segment',
                 labels={'rfm_segment':'','rfm_score':'RFM Score (3–15)'})
    fig.update_layout(showlegend=False, height=320)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Profile Table")
    profile = (
        rfm[rfm['rfm_segment'].isin(RFM_ORDER)]
        .groupby('rfm_segment')
        .agg(
            Customers     = ('household_key',  'count'),
            Avg_Spend     = ('total_spend',    'mean'),
            Avg_Baskets   = ('n_baskets',      'mean'),
            Avg_Basket    = ('avg_basket',     'mean'),
            Recency_Days  = ('recency_days',   'mean'),
            Churn_Rate    = ('is_churned',     'mean'),
        )
        .reindex(RFM_ORDER).round(1).reset_index()
    )
    profile.columns = ['Segment','Customers','Avg 2yr Spend ($)',
                        'Avg Baskets','Avg Basket ($)','Avg Recency (days)','Churn Rate']
    profile['Customers']        = profile['Customers'].apply(lambda x: f'{int(x):,}')
    profile['Avg 2yr Spend ($)']= profile['Avg 2yr Spend ($)'].apply(lambda x: f'${x:.0f}')
    profile['Avg Basket ($)']   = profile['Avg Basket ($)'].apply(lambda x: f'${x:.2f}')
    profile['Churn Rate']       = profile['Churn Rate'].apply(lambda x: f'{x:.1%}')
    st.dataframe(profile, hide_index=True, use_container_width=True)

    st.subheader("Recommended Action by Segment")
    action_df = pd.DataFrame({
        'Segment' : RFM_ORDER,
        'Campaign': ['TypeB mailer', 'TypeB mailer', 'TypeA mailer', 'No spend'],
        'Reason'  : [
            'High loyalty — TypeA overkill, TypeB sufficient; protect margin',
            'Reliable shoppers — smaller incentive converts them',
            'Declining engagement — highest-value campaign justified by churn risk',
            'CLV too low — mailer cost exceeds expected incremental return',
        ],
        'Priority': ['Medium','Medium','HIGH','Low'],
    })
    st.dataframe(action_df, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CAMPAIGN ROI
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Campaign A/B Test & ROI Analysis")
    st.warning(
        "⚠️ **Methodological limitation:** This analysis uses a within-household pre/post "
        "comparison — spend during the campaign window vs the 8 weeks prior. "
        "**There is no true control group.** Spend may have risen organically due to seasonality, "
        "store promotions, or unrelated household events. Treat these as upper-bound ROI estimates, "
        "not causal proof of campaign effectiveness."
    )
    st.markdown(
        "**Method:** within-household pre/post comparison. "
        "For each household in a campaign, we compare their average weekly spend "
        "**during** the campaign window vs the **8 weeks before** it started. "
        "Uplift = during − before. Mailer costs from **DMA 2023** industry benchmarks."
    )

    c1,c2,c3 = st.columns(3)
    best = camp_roi.iloc[0]
    c1.metric(f"{best['campaign_type']} Revenue ROI",
              f"{best['revenue_roi']:.2f}x",
              f"${best['avg_uplift_weekly']:+.2f}/wk uplift per HH")
    c2.metric("Total Incremental Revenue",
              f"${camp_roi['total_incremental_rev'].sum():,.0f}",
              "Across all campaigns, all types")
    c3.metric("Total Campaign Households",
              f"{camp_roi['n_households'].sum():,}")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        fig = px.bar(camp_roi, x='campaign_type', y='avg_uplift_weekly',
                     color='campaign_type', color_discrete_map=CAMP_COLORS,
                     title='Avg Weekly Spend Uplift per Household ($)<br>'
                           '<sup>* = p<0.05 | ** = p<0.01 | *** = p<0.001</sup>',
                     labels={'campaign_type':'Campaign Type',
                             'avg_uplift_weekly':'Weekly Uplift ($/HH)'},
                     text='avg_uplift_weekly')
        fig.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
        fig.add_hline(y=0, line_dash='dash', line_color='black')
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig = go.Figure()
        fig.add_bar(name='Mailer Cost ($)',
                    x=camp_roi['campaign_type'], y=camp_roi['total_mailer_cost'],
                    marker_color='#e74c3c', opacity=0.85)
        fig.add_bar(name='Incremental Revenue ($)',
                    x=camp_roi['campaign_type'], y=camp_roi['total_incremental_rev'],
                    marker_color='#27ae60', opacity=0.85)
        fig.update_layout(barmode='group', title='Total Cost vs Incremental Revenue',
                          yaxis_title='Amount ($)', height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Campaign ROI Summary Table")
    roi_display = camp_roi[[
        'campaign_type','n_households','avg_pre_weekly_spend','avg_during_weekly_spend',
        'avg_uplift_weekly','total_incremental_rev','total_mailer_cost',
        'revenue_roi','net_roi','p_value','significant'
    ]].copy()
    roi_display.columns = [
        'Type','Households','Pre $/wk','During $/wk',
        'Uplift $/wk','Total Uplift ($)','Mailer Cost ($)',
        'Revenue ROI','Net ROI (margin adj.)','p-value','Significant'
    ]
    roi_display['Households']          = roi_display['Households'].apply(lambda x: f'{int(x):,}')
    roi_display['Pre $/wk']            = roi_display['Pre $/wk'].apply(lambda x: f'${x:.2f}')
    roi_display['During $/wk']         = roi_display['During $/wk'].apply(lambda x: f'${x:.2f}')
    roi_display['Uplift $/wk']         = roi_display['Uplift $/wk'].apply(lambda x: f'${x:+.2f}')
    roi_display['Total Uplift ($)']    = roi_display['Total Uplift ($)'].apply(lambda x: f'${x:,.0f}')
    roi_display['Mailer Cost ($)']     = roi_display['Mailer Cost ($)'].apply(lambda x: f'${x:,.0f}')
    roi_display['Revenue ROI']         = roi_display['Revenue ROI'].apply(lambda x: f'{x:.2f}x')
    roi_display['Net ROI (margin adj.)'] = roi_display['Net ROI (margin adj.)'].apply(lambda x: f'{x:.2f}x')
    roi_display['p-value']             = roi_display['p-value'].apply(lambda x: f'{x:.4f}')
    st.dataframe(roi_display, hide_index=True, use_container_width=True)

    st.markdown(
        "_Mailer costs: TypeA=$1.50, TypeB=$0.80, TypeC=$0.50 per household "
        "(DMA 2023 industry benchmarks — stated assumption). "
        "Net ROI applies 27% grocery margin (FMI 2023)._"
    )

    # Uplift distribution
    if not uplift.empty:
        st.subheader("Uplift Distribution by Campaign Type")
        fig = px.box(uplift, x='campaign_type', y='uplift_weekly',
                     color='campaign_type', color_discrete_map=CAMP_COLORS,
                     title='Weekly Spend Uplift Distribution per Household',
                     labels={'campaign_type':'Campaign Type',
                             'uplift_weekly':'Weekly Uplift ($/HH)'},
                     category_orders={'campaign_type':['TypeA','TypeB','TypeC']})
        fig.add_hline(y=0, line_dash='dash', line_color='black')
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CLV PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Customer Lifetime Value — XGBoost + Log-Transform")
    st.markdown(
        "**17 Year-1 behavioural + demographic features** → **log(Year-2 spend)**, back-transformed.  \n"
        "Trained on 80% of households, validated on a held-out 20% test set.  \n"
        "Log-transform forces proportional accuracy across the full spend range ($700–$7,000+)."
    )

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("MAE vs Naive Baseline",
              f"{clv_m.get('rf_improvement_pct',0):.1f}% better",
              f"XGB ${clv_m.get('mae',0):.0f}  vs  Naive ${clv_m.get('naive_mae',0):.0f}",
              help="Naive baseline = predict Year-2 spend = Year-1 spend")
    c2.metric("Median Abs Error",        f"${clv_m.get('median_ae',0):.2f}",
              help="Median absolute error on 20% held-out test set")
    c3.metric("R²",                      f"{clv_m.get('r2',0):.3f}",
              help="Variance explained — 1.0 = perfect, 0.0 = no better than predicting the mean")
    c4.metric("Top-20% Precision",       f"{clv_m.get('top20_precision',0)*100:.0f}%",
              help="% of households the model ranked as top-spenders who actually were")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        test_set = clv[clv['in_test_set'] == 1] if 'in_test_set' in clv.columns else clv
        fig = px.scatter(test_set, x='predicted_clv', y='actual_y2_spend',
                         opacity=0.5, color_discrete_sequence=['#2980b9'],
                         title='Predicted vs Actual Year-2 Spend (test set)<br>'
                               '<sup>Each point = one household</sup>',
                         labels={'predicted_clv':'Predicted CLV ($)',
                                 'actual_y2_spend':'Actual Year-2 Spend ($)'})
        lim = max(test_set['predicted_clv'].max(), test_set['actual_y2_spend'].max())
        fig.add_shape(type='line', x0=0, y0=0, x1=lim, y1=lim,
                      line=dict(color='red', dash='dash', width=1))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        if not feat_imp.empty:
            fig = px.bar(feat_imp.head(10), x='importance', y='feature',
                         orientation='h',
                         color='importance', color_continuous_scale='Blues',
                         title='Feature Importances — What Drives CLV?',
                         labels={'importance':'Importance','feature':'Feature'})
            fig.update_layout(height=400, showlegend=False,
                              yaxis={'categoryorder':'total ascending'},
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        fig = px.histogram(clv, x='predicted_clv', nbins=40,
                           color_discrete_sequence=['#8e44ad'],
                           title='Predicted CLV Distribution (all households)',
                           labels={'predicted_clv':'Predicted CLV ($)'})
        fig.add_vline(x=clv['predicted_clv'].median(), line_dash='dash',
                      line_color='red',
                      annotation_text=f"Median ${clv['predicted_clv'].median():.0f}")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        clv_seg = rfm.groupby('rfm_segment')['predicted_clv'].mean().reindex(RFM_ORDER)
        fig = px.bar(clv_seg.reset_index(), x='rfm_segment', y='predicted_clv',
                     color='rfm_segment', color_discrete_map=RFM_COLORS,
                     title='Avg Predicted CLV by RFM Segment',
                     labels={'rfm_segment':'Segment','predicted_clv':'Avg Predicted CLV ($)'},
                     category_orders={'rfm_segment': RFM_ORDER},
                     text_auto='$.0f')
        fig.update_layout(showlegend=False, height=340)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 20 Households by Predicted CLV")
    top20 = (
        clv.nlargest(20, 'predicted_clv')[
            ['household_key','predicted_clv','actual_y2_spend','abs_error','in_test_set']
        ].copy()
    )
    top20.columns = ['Household','Predicted CLV ($)','Actual Y2 Spend ($)','Abs Error ($)','In Test Set']
    top20['Predicted CLV ($)']   = top20['Predicted CLV ($)'].apply(lambda x: f'${x:.0f}')
    top20['Actual Y2 Spend ($)'] = top20['Actual Y2 Spend ($)'].apply(lambda x: f'${x:.0f}')
    top20['Abs Error ($)']       = top20['Abs Error ($)'].apply(lambda x: f'${x:.0f}')
    st.dataframe(top20, hide_index=True, use_container_width=True)

    with st.expander("Model methodology"):
        n_folds = int(clv_m.get('n_folds', 5))
        st.markdown(f"""
**Algorithm:** XGBoost Regressor — Optuna-tuned hyperparameters, {n_folds}-fold cross-validation

**Tuning approach**
50 Optuna trials (TPE Bayesian sampler) search over: learning_rate, max_depth, subsample,
colsample_bytree, min_child_weight, reg_lambda, reg_alpha, gamma.
Each trial is scored by {n_folds}-fold CV MAE with early stopping per fold.
The best trial's parameters are then used for the final OOF evaluation and model training.

**Cross-validation approach**
Each fold trains on 80% of that fold's data (with an internal 15% early-stopping slice)
and predicts on the held-out 20%. Out-of-fold (OOF) predictions cover every household
exactly once — giving unbiased metrics without a separate held-out set.
The final model is retrained on all households using the average best iteration across folds.

**Features (17 total — all from Year 1 only):**

| Feature | What it captures |
|---|---|
| y1_total_spend | Overall spend level |
| y1_n_baskets | Purchase frequency |
| y1_avg_basket / std / cv | Basket size and consistency |
| y1_last4wk_spend | Most recent behaviour (strongest signal) |
| y1_spend_trend | Growth trajectory (h2/h1 ratio) |
| y1_disc_ratio | Promotion sensitivity |
| y1_recency / tenure | Engagement timing |
| n_departments | Product diversity |
| campaign_engaged / y1_redemptions | Marketing receptiveness |
| income_val / age_val | Demographic capacity |

**CV Results ({n_folds}-fold)**
- Households: {clv_m.get('n_train', 0):,} (all used — OOF covers 100%)
- MAE: ${clv_m.get('mae',0):.2f} ± ${clv_m.get('mae_std',0):.2f} | Median AE: ${clv_m.get('median_ae',0):.2f}
- R²: {clv_m.get('r2',0):.3f} ± {clv_m.get('r2_std',0):.3f}
- Top-20% Precision: {clv_m.get('top20_precision',0)*100:.0f}%

**Baseline comparison**
- Naive baseline: predict Year-2 spend = Year-1 spend → MAE ${clv_m.get('naive_mae',0):.2f}
- XGBoost (CV) → MAE ${clv_m.get('mae',0):.2f} ({clv_m.get('rf_improvement_pct',0):.1f}% improvement)
- The scatter plot shows OOF predictions — every point is a true out-of-sample prediction

**Why XGBoost over BG/NBD here:**
BG/NBD assumes customers can permanently "drop out." Grocery loyalty card holders
rarely stop shopping permanently — they switch stores. This assumption violation
causes BG/NBD to over-predict for high-frequency customers. XGBoost makes
no distributional assumptions and directly learns from observed patterns.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CHURN RISK
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Churn Prediction — Logistic Regression")
    st.markdown(
        "**Definition:** churned = no purchase in the last 90 days of the dataset.  \n"
        "**Features:** recency, frequency, spend, tenure, campaign engagement, "
        "product diversity, income, age.  \n"
        "**Train/test split:** 80/20 stratified."
    )

    overall_churn = rfm['is_churned'].mean() if 'is_churned' in rfm.columns else churn['is_churned'].mean()
    n_high_risk   = int((churn['churn_probability'] >= 0.70).sum())
    rev_at_risk   = churn[churn['churn_probability'] >= 0.70]['predicted_clv'].sum()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Overall Churn Rate",       f"{overall_churn*100:.1f}%")
    c2.metric("High-Risk Households",     f"{n_high_risk:,}",
              help="Churn probability ≥ 70%")
    c3.metric("Revenue at Risk",          f"${rev_at_risk:,.0f}",
              help="Sum of predicted CLV for high-risk households")
    c4.metric("Revenue at Risk per HH",
              f"${rev_at_risk/n_high_risk:.0f}" if n_high_risk > 0 else "$0")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        fig = px.histogram(churn, x='churn_probability', nbins=30,
                           color_discrete_sequence=['#e74c3c'],
                           title='Churn Probability Distribution',
                           labels={'churn_probability':'Churn Probability'})
        fig.add_vline(x=0.70, line_dash='dash', line_color='black',
                      annotation_text='High-risk threshold (0.70)')
        fig.update_layout(height=370)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        churn_by_seg = churn.groupby('rfm_segment')['is_churned'].mean().reindex(RFM_ORDER)
        fig = px.bar(churn_by_seg.reset_index(), x='rfm_segment', y='is_churned',
                     color='rfm_segment', color_discrete_map=RFM_COLORS,
                     title='Churn Rate by RFM Segment',
                     labels={'rfm_segment':'Segment','is_churned':'Churn Rate'},
                     category_orders={'rfm_segment': RFM_ORDER},
                     text_auto='.1%')
        fig.update_layout(showlegend=False, height=370,
                          yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)

    # High-risk customers table
    st.subheader(f"High-Risk Households (churn prob ≥ 70%) — {n_high_risk:,} total")
    high_risk_tbl = (
        churn[churn['churn_probability'] >= 0.70]
        .sort_values('predicted_clv', ascending=False)
        .head(20)[['household_key','churn_probability','predicted_clv','rfm_segment']]
    )
    high_risk_tbl.columns = ['Household','Churn Prob','Predicted CLV ($)','Segment']
    high_risk_tbl['Churn Prob']      = high_risk_tbl['Churn Prob'].apply(lambda x: f'{x:.1%}')
    high_risk_tbl['Predicted CLV ($)'] = high_risk_tbl['Predicted CLV ($)'].apply(lambda x: f'${x:.2f}')
    st.dataframe(high_risk_tbl, hide_index=True, use_container_width=True)

    st.subheader("Re-engagement Economics")
    col_a, col_b = st.columns(2)
    with col_a:
        mailer_cost  = st.number_input("Mailer cost per household ($)", 0.5, 5.0, 1.50, 0.1)
        conv_rate    = st.slider("Assumed re-engagement conversion rate (%)", 1, 30, 5) / 100
    with col_b:
        saved_rev    = rev_at_risk * conv_rate
        total_cost   = n_high_risk * mailer_cost
        re_eng_roi   = saved_rev / total_cost if total_cost > 0 else 0
        st.metric("Saved revenue (at conversion rate)",  f"${saved_rev:,.0f}")
        st.metric("Total campaign cost",                 f"${total_cost:,.0f}")
        st.metric("Re-engagement ROI",                   f"{re_eng_roi:.2f}x")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — TRENDS & SEASONALITY
# ══════════════════════════════════════════════════════════════════════════════
INCOME_ORDER = [
    'Under 15K','15-24K','25-34K','35-49K','50-74K',
    '75-99K','100-124K','125-149K','150-174K','175-199K','200-249K','250K+',
]
AGE_ORDER = ['19-24','25-34','35-44','45-54','55-64','65+']

with tab6:
    st.subheader("Spending Trends & Seasonality")

    # Weekly spend — annotated with campaign periods
    fig = px.line(wkly, x='week_no', y='total_spend',
                  title='Total Weekly Spend — shaded bands show active campaign periods',
                  labels={'week_no': 'Week', 'total_spend': 'Total Spend ($)'})
    fig.add_vline(x=52, line_dash='dash', line_color='grey',
                  annotation_text='Year 2 start')
    if not camp_wks.empty:
        _camp_colors = {'TypeA': '#e74c3c', 'TypeB': '#f39c12', 'TypeC': '#3498db'}
        _seen = set()
        for _, cw in camp_wks.iterrows():
            ctype  = cw['description']
            colour = _camp_colors.get(ctype, '#888888')
            label  = ctype if ctype not in _seen else None
            _seen.add(ctype)
            fig.add_vrect(
                x0=cw['start_week'], x1=cw['end_week'],
                fillcolor=colour, opacity=0.12, layer='below', line_width=0,
                annotation_text=label or '',
                annotation_position='top left',
                annotation_font_size=9,
            )
    fig.update_layout(height=340)
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.line(wkly, x='week_no', y='avg_basket',
                      title='Avg Basket Size by Week ($)',
                      labels={'week_no': 'Week', 'avg_basket': 'Avg Basket ($)'})
        fig.add_vline(x=52, line_dash='dash', line_color='grey')
        fig.update_layout(height=290)
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        fig = px.line(wkly, x='week_no', y='n_households',
                      title='Active Households per Week',
                      labels={'week_no': 'Week', 'n_households': 'Households'})
        fig.add_vline(x=52, line_dash='dash', line_color='grey')
        fig.update_layout(height=290)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### Monthly Spend by RFM Segment")
    mthly_filt = mthly_seg[mthly_seg['rfm_segment'].isin(RFM_ORDER)]
    fig = px.line(mthly_filt, x='cal_month', y='total_spend',
                  color='rfm_segment', color_discrete_map=RFM_COLORS,
                  title='Monthly Revenue by Segment',
                  labels={'cal_month': 'Month', 'total_spend': 'Total Spend ($)',
                          'rfm_segment': 'Segment'},
                  category_orders={'rfm_segment': RFM_ORDER})
    fig.update_layout(height=340, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### Cohort Retention Matrix")
    st.caption("% of each monthly cohort that made at least one purchase N months after joining")
    if not retention.empty:
        ret_pivot = retention.pivot(
            index='cohort_month', columns='months_since', values='retention_pct'
        )
        cols_show = [c for c in ret_pivot.columns if c <= 12]
        ret_show  = ret_pivot[cols_show]
        fig = go.Figure(data=go.Heatmap(
            z=ret_show.values,
            x=[f'M+{c}' for c in ret_show.columns],
            y=[str(r) for r in ret_show.index],
            colorscale='Blues', zmin=0, zmax=100,
            text=[[f'{v:.0f}%' if not pd.isna(v) else '' for v in row]
                  for row in ret_show.values],
            texttemplate='%{text}',
        ))
        fig.update_layout(
            title='Monthly Cohort Retention (%)',
            xaxis_title='Months since first purchase',
            yaxis_title='Cohort',
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### Customer Acquisition")
    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.bar(acq, x='first_cal_month', y='n_new',
                     title='New Customers per Month',
                     labels={'first_cal_month': 'Month', 'n_new': 'New Customers'},
                     color_discrete_sequence=['#2980b9'])
        fig.update_layout(height=290, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        fig = px.line(acq, x='first_cal_month', y='avg_clv',
                      title='Avg Predicted CLV of New Customers ($)',
                      labels={'first_cal_month': 'Month', 'avg_clv': 'Avg CLV ($)'},
                      markers=True, color_discrete_sequence=['#27ae60'])
        fig.update_layout(height=290, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    if not acq_seg.empty:
        acq_filt = acq_seg[acq_seg['rfm_segment'].isin(RFM_ORDER)]
        fig = px.bar(acq_filt, x='first_cal_month', y='n_new',
                     color='rfm_segment', color_discrete_map=RFM_COLORS,
                     barmode='stack',
                     title='New Customers by Eventual RFM Segment',
                     labels={'first_cal_month': 'Month', 'n_new': 'New Customers',
                             'rfm_segment': 'Segment'},
                     category_orders={'rfm_segment': RFM_ORDER})
        fig.update_layout(height=310, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.subheader("Customer Demographics Analysis")

    col_l, col_r = st.columns(2)
    with col_l:
        if not rfm_inc.empty:
            inc_f = rfm_inc[rfm_inc['rfm_segment'].isin(RFM_ORDER)].copy()
            inc_f['income_desc'] = pd.Categorical(
                inc_f['income_desc'], categories=INCOME_ORDER, ordered=True
            )
            inc_f = inc_f.dropna(subset=['income_desc']).sort_values('income_desc')
            fig = px.bar(inc_f, x='income_desc', y='n_households',
                         color='rfm_segment', color_discrete_map=RFM_COLORS,
                         barmode='stack',
                         title='Income Distribution by Segment',
                         labels={'income_desc': 'Income', 'n_households': 'Households',
                                 'rfm_segment': 'Segment'},
                         category_orders={'income_desc': INCOME_ORDER,
                                          'rfm_segment': RFM_ORDER})
            fig.update_layout(height=370, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    with col_r:
        if not rfm_age.empty:
            age_f = rfm_age[rfm_age['rfm_segment'].isin(RFM_ORDER)].copy()
            age_f['age_desc'] = pd.Categorical(
                age_f['age_desc'], categories=AGE_ORDER, ordered=True
            )
            age_f = age_f.dropna(subset=['age_desc']).sort_values('age_desc')
            fig = px.bar(age_f, x='age_desc', y='n_households',
                         color='rfm_segment', color_discrete_map=RFM_COLORS,
                         barmode='stack',
                         title='Age Distribution by Segment',
                         labels={'age_desc': 'Age Group', 'n_households': 'Households',
                                 'rfm_segment': 'Segment'},
                         category_orders={'age_desc': AGE_ORDER,
                                          'rfm_segment': RFM_ORDER})
            fig.update_layout(height=370, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    col_l, col_r = st.columns(2)
    with col_l:
        if not rfm_hh_df.empty:
            hh_f = rfm_hh_df[rfm_hh_df['rfm_segment'].isin(RFM_ORDER)]
            fig = px.bar(hh_f, x='hh_comp_desc', y='n_households',
                         color='rfm_segment', color_discrete_map=RFM_COLORS,
                         barmode='group',
                         title='Household Composition by Segment',
                         labels={'hh_comp_desc': 'Household Type',
                                 'n_households': 'Households', 'rfm_segment': 'Segment'},
                         category_orders={'rfm_segment': RFM_ORDER})
            fig.update_layout(height=370, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
    with col_r:
        if not clv_inc.empty:
            clv_inc_c = clv_inc.copy()
            clv_inc_c['income_desc'] = pd.Categorical(
                clv_inc_c['income_desc'], categories=INCOME_ORDER, ordered=True
            )
            clv_inc_c = clv_inc_c.dropna(subset=['income_desc']).sort_values('income_desc')
            fig = px.bar(clv_inc_c, x='income_desc', y='avg_clv',
                         title='Avg Predicted CLV by Income Bracket ($)',
                         labels={'income_desc': 'Income', 'avg_clv': 'Avg CLV ($)'},
                         color='avg_clv', color_continuous_scale='Greens',
                         text_auto='$.0f')
            fig.update_layout(height=370, xaxis_tickangle=-45,
                               coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Segment Composition by Income (% of income bracket)")
    if not rfm_inc.empty:
        inc_piv = rfm_inc[rfm_inc['rfm_segment'].isin(RFM_ORDER)].pivot_table(
            index='income_desc', columns='rfm_segment',
            values='n_households', aggfunc='sum', fill_value=0,
        )
        inc_piv = inc_piv.div(inc_piv.sum(axis=1), axis=0).mul(100).round(1)
        avail   = [i for i in INCOME_ORDER if i in inc_piv.index]
        inc_piv = inc_piv.reindex(avail)
        seg_cols = [s for s in RFM_ORDER if s in inc_piv.columns]
        fig = go.Figure(data=go.Heatmap(
            z=inc_piv[seg_cols].values,
            x=seg_cols,
            y=list(inc_piv.index),
            colorscale='Blues',
            text=[[f'{v:.0f}%' for v in row] for row in inc_piv[seg_cols].values],
            texttemplate='%{text}',
        ))
        fig.update_layout(title='% of Income Bracket in Each Segment', height=370,
                          xaxis_title='Segment', yaxis_title='Income Bracket')
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — PRODUCT ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab8:
    st.subheader("Product & Category Analytics")

    st.markdown("#### Brand Loyalty by Segment")
    if not brand_df.empty:
        brand_f = brand_df[brand_df['rfm_segment'].isin(RFM_ORDER)]
        fig = px.bar(brand_f, x='rfm_segment', y='pct',
                     color='brand',
                     barmode='stack',
                     title='Revenue Split: National vs Private Label by Segment (%)',
                     labels={'rfm_segment': 'Segment', 'pct': '% of Spend',
                             'brand': 'Brand'},
                     category_orders={'rfm_segment': RFM_ORDER},
                     text_auto='.0f')
        fig.update_layout(height=370)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### Category Affinity by Segment")
    st.caption("Avg spend per household per department — shows where each segment concentrates their shopping")
    if not cat_df.empty:
        top_depts = cat_df.groupby('department')['total_spend'].sum().nlargest(10).index
        cat_top   = cat_df[cat_df['rfm_segment'].isin(RFM_ORDER) &
                           cat_df['department'].isin(top_depts)]
        fig = px.bar(cat_top, x='department', y='avg_spend_per_hh',
                     color='rfm_segment', color_discrete_map=RFM_COLORS,
                     barmode='group',
                     title='Avg Spend per Household by Department & Segment ($)',
                     labels={'department': 'Department',
                             'avg_spend_per_hh': 'Avg Spend/HH ($)',
                             'rfm_segment': 'Segment'},
                     category_orders={'rfm_segment': RFM_ORDER})
        fig.update_layout(height=390, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Category Spend Heatmap (Avg $/HH)")
        cat_piv = cat_top.pivot_table(
            index='department', columns='rfm_segment',
            values='avg_spend_per_hh', fill_value=0,
        )
        seg_cols = [s for s in RFM_ORDER if s in cat_piv.columns]
        cat_piv  = cat_piv[seg_cols]
        fig = go.Figure(data=go.Heatmap(
            z=cat_piv.values,
            x=list(cat_piv.columns),
            y=list(cat_piv.index),
            colorscale='Blues',
            text=[[f'${v:.0f}' for v in row] for row in cat_piv.values],
            texttemplate='%{text}',
        ))
        fig.update_layout(title='Avg Spend per HH by Department & Segment',
                          height=400, xaxis_title='Segment',
                          yaxis_title='Department')
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### Discount Sensitivity by Segment")
    st.caption(
        "Households bucketed by how much of their Year-1 spend came from retail promotions. "
        "Shows whether heavy discounters are actually better or worse long-term customers."
    )
    if not disc_sens.empty:
        BUCKET_ORDER = ['<5%', '5-10%', '10-15%', '15-20%', '20%+']
        disc_f = disc_sens[disc_sens['rfm_segment'].isin(RFM_ORDER)].copy()
        disc_f['disc_bucket'] = pd.Categorical(
            disc_f['disc_bucket'], categories=BUCKET_ORDER, ordered=True
        )
        disc_f = disc_f.sort_values('disc_bucket')

        col_l, col_r = st.columns(2)
        with col_l:
            fig = px.line(disc_f, x='disc_bucket', y='avg_2yr_spend',
                          color='rfm_segment', color_discrete_map=RFM_COLORS,
                          markers=True,
                          title='Avg 2-Year Spend by Discount Depth & Segment ($)',
                          labels={'disc_bucket': 'Discount Depth (Y1)',
                                  'avg_2yr_spend': 'Avg 2yr Spend ($)',
                                  'rfm_segment': 'Segment'},
                          category_orders={'rfm_segment': RFM_ORDER,
                                           'disc_bucket': BUCKET_ORDER})
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)
        with col_r:
            fig = px.line(disc_f, x='disc_bucket', y='avg_clv',
                          color='rfm_segment', color_discrete_map=RFM_COLORS,
                          markers=True,
                          title='Avg Predicted CLV by Discount Depth & Segment ($)',
                          labels={'disc_bucket': 'Discount Depth (Y1)',
                                  'avg_clv': 'Avg Predicted CLV ($)',
                                  'rfm_segment': 'Segment'},
                          category_orders={'rfm_segment': RFM_ORDER,
                                           'disc_bucket': BUCKET_ORDER})
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)

        # Customer count heatmap: how many HH per segment × discount bucket
        disc_piv = disc_f.pivot_table(
            index='rfm_segment', columns='disc_bucket',
            values='n_customers', fill_value=0,
        )
        bucket_cols = [b for b in BUCKET_ORDER if b in disc_piv.columns]
        disc_piv    = disc_piv.reindex(
            index=[s for s in RFM_ORDER if s in disc_piv.index],
            columns=bucket_cols,
        )
        fig = go.Figure(data=go.Heatmap(
            z=disc_piv.values,
            x=list(disc_piv.columns),
            y=list(disc_piv.index),
            colorscale='Blues',
            text=[[f'{int(v):,}' for v in row] for row in disc_piv.values],
            texttemplate='%{text}',
        ))
        fig.update_layout(
            title='Household Count by Segment × Discount Bucket',
            xaxis_title='Discount Depth (Y1)',
            yaxis_title='Segment',
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — SQL EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab9:
    st.subheader("SQL Explorer")
    st.markdown("Query the underlying SQLite database directly.")

    PRESETS = {
        "Top 20 customers by predicted CLV": """
SELECT r.household_key, r.rfm_segment, r.total_spend,
       c.predicted_clv, c.actual_y2_spend,
       ROUND(c.predicted_clv - c.actual_y2_spend, 2) AS prediction_error
FROM rfm_segments r
JOIN clv_predictions c ON r.household_key = c.household_key
ORDER BY c.predicted_clv DESC
LIMIT 20;
""",
        "Campaign ROI by type": """
SELECT campaign_type, n_households,
       ROUND(avg_pre_weekly_spend, 2)    AS pre_weekly_spend,
       ROUND(avg_during_weekly_spend, 2) AS during_weekly_spend,
       ROUND(avg_uplift_weekly, 2)       AS uplift_per_week,
       ROUND(total_incremental_rev, 0)   AS total_uplift,
       ROUND(revenue_roi, 2)             AS revenue_roi,
       ROUND(net_roi, 2)                 AS net_roi_margin_adj,
       p_value,
       significant
FROM campaign_roi
ORDER BY revenue_roi DESC;
""",
        "High churn-risk customers with revenue at stake": """
SELECT cs.household_key, cs.churn_probability, cs.predicted_clv,
       cs.rfm_segment, r.total_spend, r.n_baskets, r.recency_days
FROM churn_scores cs
JOIN rfm_segments r ON cs.household_key = r.household_key
WHERE cs.churn_probability >= 0.70
ORDER BY cs.predicted_clv DESC
LIMIT 30;
""",
        "RFM segment summary with avg CLV": """
SELECT rfm_segment,
       COUNT(*)                          AS n_customers,
       ROUND(AVG(total_spend), 2)        AS avg_2yr_spend,
       ROUND(AVG(n_baskets), 1)          AS avg_baskets,
       ROUND(AVG(recency_days), 0)       AS avg_recency_days,
       ROUND(AVG(predicted_clv), 2)      AS avg_predicted_clv,
       ROUND(AVG(is_churned)*100, 1)     AS churn_rate_pct
FROM rfm_segments
GROUP BY rfm_segment
ORDER BY avg_2yr_spend DESC;
""",
        "Cohort retention by join quarter": """
SELECT first_quarter,
       n_customers,
       ROUND(avg_spend, 2)      AS avg_2yr_spend,
       ROUND(avg_baskets, 1)    AS avg_baskets,
       ROUND(avg_clv, 2)        AS avg_predicted_clv,
       ROUND(churn_rate*100, 1) AS churn_rate_pct
FROM cohort_results
ORDER BY first_quarter;
""",
        "Financial scenario comparison": """
SELECT scenario, n_households,
       mailer_cost, incremental_revenue,
       revenue_roi, cost_saving
FROM financial_scenarios
ORDER BY revenue_roi DESC;
""",
        "Top spending categories": """
SELECT department, n_households, total_spend,
       ROUND(avg_item_spend, 2) AS avg_item_spend,
       n_baskets
FROM category_spend
ORDER BY total_spend DESC
LIMIT 10;
""",
        "Campaign vs organic buyer comparison": """
SELECT buyer_type,
       COUNT(*)                       AS n_households,
       ROUND(AVG(total_spend), 2)     AS avg_spend,
       ROUND(AVG(n_baskets), 1)       AS avg_baskets
FROM buyer_segments
GROUP BY buyer_type;
""",
    }

    preset    = st.selectbox("Choose a preset query", ['-- Custom SQL --'] + list(PRESETS.keys()))
    default   = PRESETS.get(preset, "SELECT * FROM rfm_segments LIMIT 10;")
    sql_input = st.text_area("SQL Query", value=default, height=180)

    col_run, col_schema = st.columns([1, 5])
    run = col_run.button("Run Query", type="primary")
    col_schema.caption(
        "Raw: transactions · households · campaign_assignments · campaign_descriptions · "
        "coupons · coupon_redemptions · products  |  "
        "Views: customer_summary · campaign_summary · weekly_spend · category_spend · buyer_segments  |  "
        "Results: rfm_segments · clv_predictions · churn_scores · campaign_roi · "
        "cohort_results · financial_scenarios · clv_metrics  |  "
        "New: cohort_retention · weekly_trend · monthly_segment_spend · "
        "demographics_income · demographics_age · demographics_hh_comp · clv_by_income · "
        "brand_by_segment · category_by_segment · acquisition_trend · acquisition_by_segment"
    )

    if run:
        try:
            result = query(sql_input)
            st.success(f"{len(result):,} rows returned")
            st.dataframe(result, use_container_width=True, height=420)
            st.download_button("Download CSV",
                               data=result.to_csv(index=False).encode(),
                               file_name="query_result.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"SQL error: {exc}")

    with st.expander("Database schema"):
        st.markdown("""
**Raw tables** (from CSVs):

| Table | Key columns |
|---|---|
| `transactions` | household_key, basket_id, day, week_no, product_id, quantity, sales_value |
| `households` | household_key, age_desc, income_desc, marital_status_code, hh_comp_desc |
| `campaign_assignments` | description (TypeA/B/C), household_key, campaign |
| `campaign_descriptions` | campaign, description, start_day, end_day |
| `products` | product_id, department, brand, commodity_desc |
| `coupons` | coupon_upc, product_id, campaign |
| `coupon_redemptions` | household_key, day, coupon_upc, campaign |

**Views:**

| View | What it answers |
|---|---|
| `customer_summary` | Per-HH: total spend, baskets, recency, tenure |
| `campaign_summary` | HH count per campaign type |
| `weekly_spend` | Weekly spend per household |
| `category_spend` | Revenue by department |
| `buyer_segments` | Campaign enrolled vs organic |

**Analysis results:**

| Table | Contents |
|---|---|
| `rfm_segments` | RFM scores, segment, CLV, churn prob per household |
| `clv_predictions` | Predicted vs actual Y2 spend, MAE per household |
| `churn_scores` | Churn probability, label, segment per household |
| `campaign_roi` | Uplift, cost, ROI per campaign type |
| `cohort_results` | Retention & spend by join quarter |
| `financial_scenarios` | Blanket vs Targeted vs CLV-Gated ROI |
| `clv_metrics` | Overall model accuracy (MAE, RMSE, precision) |
        """)
