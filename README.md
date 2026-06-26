# Grocery Customer Analytics — Dunnhumby Complete Journey

**Live dashboard →** [grocery-customer-analysis-bj7ix8veh7rqn8sx9ssg9h.streamlit.app](https://grocery-customer-analysis-bj7ix8veh7rqn8sx9ssg9h.streamlit.app)

A full-stack customer analytics system built on 2 years of real retail transaction data (2,500 households, ~2M transactions). The goal: answer the questions a retail marketing team actually cares about — *who are our best customers, who is about to leave, and are our campaigns actually working?*

---

## Key Findings

### Customer Segments (RFM Analysis)
Customers were split into four behavioural segments based on how recently they shopped, how often, and how much they spent:

| Segment | Customers | Avg 2-Year Spend |
|---|---|---|
| Champions | 520 | $7,570 |
| Loyal | 651 | $3,750 |
| At Risk | 626 | $1,876 |
| Lost | 703 | $719 |

**Champions (21% of customers) account for a disproportionate share of revenue** — protecting this segment is the single highest-priority retention action.

### Campaign ROI — Did the Mailers Actually Work?
All three campaign types generated statistically significant uplift (p < 0.05), but their efficiency varied dramatically:

| Campaign | Revenue ROI | Net ROI (after cost) |
|---|---|---|
| TypeC | 93.6x | 25.3x |
| TypeA | 22.9x | 6.2x |
| TypeB | 12.9x | 3.5x |

**TypeC delivers 7x more revenue per dollar spent than TypeA** — yet TypeA costs almost 3x more per household. Reallocating budget toward TypeC campaigns would materially improve marketing efficiency.

### Churn Risk
- **218 households** have a churn probability ≥ 70%
- These customers represent **$111,493 in revenue at risk**
- Sending a targeted re-engagement mailer to this group costs ~$327 at TypeA rates — a small spend to protect six-figure revenue

### Cohort Retention
Customers acquired in all four quarters of 2013 showed consistent behaviour over 2 years, with churn rates between 6–7%. Campaign-engaged customers made up 60–68% of each cohort, confirming campaigns drive long-term retention, not just one-off purchases.

### Financial Targeting Model
Blanket TypeA mailers to all 2,500 households generate an 8x revenue ROI — but targeting only high-value At Risk customers with TypeA and using TypeB for Champions preserves margin without sacrificing revenue.

---

## What's Built

| Module | Description |
|---|---|
| `main.py` | End-to-end pipeline: cleans data, runs all analysis, trains ML models, saves results to Supabase |
| `dashboard.py` | Streamlit dashboard with 8 analytical sections and a live SQL explorer |
| `database.py` | SQLAlchemy + psycopg2 connection layer (Supabase PostgreSQL) |
| `migrate.py` | One-time script to seed Supabase from local SQLite results |

## Analysis Sections

1. **RFM Segmentation** — Recency / Frequency / Monetary scoring with segment profiles and recommended campaign actions per segment
2. **Campaign ROI** — Pre/post spend uplift with Welch's t-test significance testing across all campaign types and individual household-level analysis
3. **Cohort Retention** — Quarter-of-acquisition cohorts tracked across 8 periods; spend, basket frequency, and churn by cohort
4. **Customer Lifetime Value** — XGBoost regressor (Optuna-tuned, 5-fold CV) predicting 2-year CLV; R² = 0.75, top-20 precision = 76%
5. **Churn Prediction** — Logistic Regression churn scoring with re-engagement economics calculator
6. **Financial Targeting** — Scenario modelling comparing blanket vs. targeted campaign spend
7. **Demographics** — Spend and loyalty breakdown by income band, household composition, and age group
8. **SQL Explorer** — Live query interface against the Supabase database with preset analytical queries

---

## Tech Stack

- **Data & Analysis:** Python, Pandas, NumPy
- **ML:** XGBoost, Scikit-learn, Optuna (Bayesian hyperparameter tuning), Lifetimes (BG/NBD)
- **Visualisation:** Plotly, Streamlit
- **Database:** Supabase (PostgreSQL) via SQLAlchemy + psycopg2
- **Deployment:** Streamlit Cloud

---

## Running Locally

```bash
git clone https://github.com/abhaykhaiwal/Grocery-Customer-Analysis
cd Grocery-Customer-Analysis
pip install -r requirements.txt
```

Create a `.env` file with your Supabase credentials:
```
DB_HOST=your-pooler-host
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres.your-project-ref
DB_PASSWORD=your-password
```

```bash
# View the dashboard (reads from Supabase — no local data files needed)
streamlit run dashboard.py

# Re-run the full pipeline (requires the Dunnhumby CSVs from Kaggle)
python main.py
```

**Dataset:** [Dunnhumby — The Complete Journey](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey) (Kaggle)
The two large files (`transaction_data.csv`, `causal_data.csv`) are excluded from this repo due to size — download them from Kaggle and place them in the project root before running `main.py`.

---

## Project Structure

```
├── main.py              # Full analysis + ML pipeline
├── dashboard.py         # Streamlit dashboard
├── database.py          # Supabase connection layer
├── migrate.py           # SQLite → Supabase migration utility
├── requirements.txt     # Python dependencies
├── *.csv                # Small reference tables (campaigns, products, demographics)
└── .env                 # Credentials — not committed (add your own)
```
