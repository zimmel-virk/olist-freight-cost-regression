# Olist Freight Cost Regression — Data Cleaning, Feature Engineering & Ridge

A data-science project that predicts **item-level freight cost in Brazilian e-commerce** using the public Olist marketplace dataset.

This repository showcases a complete analytical workflow rather than only model fitting: **nine relational CSV files are audited, cleaned, validated, merged and transformed into a leakage-safe modelling dataset** before multiple regression models are compared.

## Why this project is portfolio-relevant

The strongest part of this project is the data work. It demonstrates practical experience with:

- multi-file relational data integration;
- dataset-level and column-level quality audits;
- candidate-key and foreign-key integrity checks;
- one-to-many / many-to-many relationship analysis;
- duplicate detection and treatment;
- missing-value analysis;
- string, numeric and datetime sanitisation;
- geolocation validation and aggregation;
- leakage prevention;
- group-aware train/test splitting and cross-validation;
- training-only imputation and preprocessing;
- outlier, influence and heteroscedasticity diagnostics;
- domain-informed feature engineering;
- regularised regression;
- order-level bootstrap uncertainty analysis.

## Problem

The target is:

```text
freight_value
```

The objective is to estimate the freight charge for an individual order item from information available at or near purchase time, including:

- product price;
- product weight and dimensions;
- seller-to-customer distance;
- product category;
- customer and seller region;
- purchase timing;
- order-level context.

## Dataset

**Brazilian E-Commerce Public Dataset by Olist**

The coursework uses nine related CSV files:

```text
olist_customers_dataset.csv
olist_geolocation_dataset.csv
olist_order_items_dataset.csv
olist_order_payments_dataset.csv
olist_order_reviews_dataset.csv
olist_orders_dataset.csv
olist_products_dataset.csv
olist_sellers_dataset.csv
product_category_name_translation.csv
```

The source dataset contains roughly 100,000 orders and more than 100,000 order-item observations. The coursework records:

| Table | Rows |
|---|---:|
| Customers | 99,441 |
| Orders | 99,441 |
| Order items | 112,650 |
| Products | 32,951 |
| Sellers | 3,095 |
| Payments | 103,886 |
| Reviews | 99,224 |
| Geolocation | 1,000,163 |
| Category translations | 71 |

The final modelling workflow uses a reproducible **9,000-row order-item sample** for coursework efficiency.

## Data-quality and preprocessing workflow

### 1. Preserve raw data

Raw DataFrames are left unchanged while separate cleaned working copies are created.

### 2. Audit every source table

The notebook records:

- row/column counts;
- duplicate rows;
- missing cells and percentages;
- memory usage;
- data types;
- column cardinality.

### 3. Validate relational integrity

Candidate keys and foreign keys are audited before merging. Relationship multiplicity is inspected so that direct joins do not accidentally multiply observations.

This is particularly important because an order may have multiple:

- order items;
- payment transactions;
- review records.

The analytical unit is intentionally defined as **one order item**.

### 4. Sanitise values

The project performs:

- whitespace stripping;
- empty-string-to-missing conversion;
- standardisation of Brazilian state codes;
- category-name normalisation;
- explicit datetime parsing;
- numeric conversion with failure auditing;
- duplicate treatment;
- physical plausibility checks.

### 5. Clean geolocation data

The geolocation table contains repeated coordinates for ZIP-code prefixes. Coordinates are validated and aggregated before joining, preventing a large many-to-many expansion.

### 6. Controlled relational merges

The workflow builds an order-item analytical table using audited many-to-one joins across orders, customers, sellers, products and cleaned geographical data.

### 7. Geographical feature engineering

Seller-to-customer straight-line distance is calculated using the **Haversine formula**.

### 8. Leakage-safe model preprocessing

Model-specific transformations are learned only from training observations:

- numerical median imputation;
- scaling;
- categorical imputation;
- one-hot encoding;
- rare-category handling;
- Ridge regularisation.

Related items from the same order are kept together through **group-aware splitting**.

## Statistical and diagnostic analysis

The project goes beyond a standard regression notebook and includes:

- central tendency and spread;
- skewness and excess kurtosis;
- IQR and modified-z outlier diagnostics;
- Pearson and Spearman correlations;
- Q-Q analysis;
- residual-vs-fitted analysis;
- actual-vs-predicted calibration;
- Breusch-Pagan heteroscedasticity testing;
- VIF multicollinearity analysis;
- leverage;
- Cook's distance;
- studentised residuals.

The sampled target distribution is strongly right-skewed. In the coursework's 9,000-row analytical sample, mean freight is about **19.70 BRL**, median freight about **16.25 BRL**, and skewness about **4.96**.

## Models

Three main regression approaches are compared.

### Baseline multiple linear regression

Reference test performance:

```text
RMSE: 11.492 BRL
MAE:   5.166 BRL
R²:    0.562
```

### Feature-engineered ordinary least squares

Feature engineering adds domain-informed non-linear and interaction terms such as:

- parcel volume;
- parcel density;
- price-to-weight relationships;
- logarithmic transformations;
- squared distance;
- squared volume;
- weight × distance;
- volume × distance.

Reference test performance:

```text
RMSE: 10.601 BRL
MAE:   4.755 BRL
R²:    0.627
```

### Tuned Ridge regression

Reference test performance:

```text
RMSE: 10.629 BRL
MAE:   4.740 BRL
R²:    0.625
```

Although engineered OLS has a slightly lower single-split RMSE, the coursework selects **Ridge** as the preferred model because it has lower MAE, lower median absolute error, stronger group-aware cross-validation behaviour and greater coefficient stability under correlated engineered features.

Relative to the baseline, Ridge reduces:

- RMSE by approximately **7.5%**
- MAE by approximately **8.3%**

## What the model learned

The strongest predictor groups are associated with:

- parcel volume;
- product weight;
- geographical distance;
- interactions between weight/volume and distance;
- same-state delivery context.

The project treats coefficients as associative rather than causal.

## Robustness analysis

The final evaluation uses **order-level bootstrap resampling** rather than ordinary row-level bootstrap, preserving dependence between items belonging to the same order.

The coursework reports that Ridge reduced RMSE versus the baseline in **96.4% of order-level bootstrap samples** and reduced MAE in every bootstrap sample.

Performance is weaker for high-cost shipments; the highest freight quartile has a reported RMSE of **19.447 BRL**, so the model is positioned as decision support for quotation and budgeting rather than fully automated pricing.

## Repository structure

```text
olist-freight-cost-regression/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── olist_freight_cost_regression.ipynb
├── src/
│   ├── __init__.py
│   └── freight_utils.py
├── data/
│   ├── raw/
│   │   └── README.md
│   └── processed/
├── results/
│   ├── README.md
│   ├── reference_model_metrics.csv
│   └── reference_rmse_comparison.png
├── figures/
└── docs/
    ├── original_coursework_report.pdf
    ├── original_notebook_export.html
    └── PORTFOLIO_NOTES.md
```

## Setup

Python 3.11 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the nine Olist CSV files in:

```text
data/raw/
```

Then open:

```text
notebooks/olist_freight_cost_regression.ipynb
```

and run the notebook from the repository root.

## Data policy

The raw Olist CSV files are intentionally **not committed** to this repository. They are third-party dataset files and are substantially larger than the source code.

The `data/raw/README.md` file lists the exact filenames expected by the notebook.

## Portfolio provenance

The notebook in this repository was reconstructed from the supplied Jupyter HTML export. Code and narrative are preserved from the coursework, while cell outputs were omitted from the `.ipynb` copy to make the repository smaller and easier to review.

The original rendered HTML and PDF are retained under `docs/` as evidence of the completed analysis.

## Skills demonstrated

**Data cleaning:** missing values, duplicates, types, plausibility checks, sanitisation  
**Data engineering:** relational joins, key integrity, multiplicity analysis, 1NF analytical table construction  
**Feature engineering:** Haversine distance, parcel metrics, log/squared/interaction terms  
**Statistics:** distributions, correlations, outliers, heteroscedasticity, multicollinearity, influence diagnostics  
**Machine learning:** linear regression, Ridge regression, group-aware cross-validation, leakage-safe pipelines  
**Evaluation:** RMSE, MAE, R², residual analysis, group bootstrap, subgroup error analysis  
**Python:** pandas, NumPy, SciPy, Matplotlib, scikit-learn, statsmodels

## Author

**Zimmel Javed Virk**  
BSc Computer Science — University of London
