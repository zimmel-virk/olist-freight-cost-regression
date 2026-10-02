# Olist Freight Cost Regression — Data Cleaning, Feature Engineering & Ridge

A data-science project that predicts **item-level freight cost in Brazilian e-commerce** using the Brazilian E-Commerce Public Dataset by Olist.

The project develops a complete analytical workflow across nine related CSV files. The data is audited, cleaned, sanitised, validated and merged into an analysis-ready order-item dataset before regression models are trained, evaluated and compared.

## Project Objective

The target variable is:

```text
freight_value
```

The objective is to estimate the freight charge for an individual product item using information available at or close to purchase time, including:

- product price
- product weight and dimensions
- seller-to-customer distance
- product category
- customer and seller state
- purchase timing
- order-level characteristics

The project also investigates which factors have the strongest relationship with freight cost and whether domain-informed feature engineering and regularisation improve predictive performance.

## Dataset

The project uses the **Brazilian E-Commerce Public Dataset by Olist**.

The source data is distributed across nine related CSV files:

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

The recorded source-table sizes are:

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

The source is relational rather than a single analysis-ready table, so a major part of the project focuses on controlled data preparation, relational integrity and preprocessing.

The final modelling workflow uses a reproducible **9,000-row order-item sample** for statistical and machine-learning analysis.

## Data Preparation and Cleaning

### Raw-data preservation

The raw DataFrames are kept unchanged while cleaned working copies are created for subsequent analysis.

### Dataset-level quality auditing

Each source table is inspected for:

- row and column counts
- duplicate rows
- missing values
- missing-value percentages
- memory usage
- data types
- column cardinality

### Relational-integrity analysis

Candidate keys and foreign keys are checked before tables are merged.

Relationship multiplicity is also analysed because an order may contain multiple:

- order items
- payment transactions
- review records

The analytical unit is intentionally defined as **one order item**.

This prevents uncontrolled many-to-many joins from artificially multiplying observations.

### Data sanitisation

The cleaning workflow includes:

- whitespace stripping
- empty-string handling
- Brazilian state-code standardisation
- category-name normalisation
- explicit datetime parsing
- numeric conversion
- conversion-failure auditing
- duplicate treatment
- missing-value analysis
- physical plausibility checks

### Geolocation processing

The geolocation dataset contains repeated coordinate records for ZIP-code prefixes.

Coordinates are validated and aggregated before joining to avoid many-to-many expansion.

Seller-to-customer geographical distance is then calculated using the **Haversine formula**.

### Controlled relational merging

The analytical dataset is constructed through audited joins across:

- orders
- order items
- customers
- sellers
- products
- cleaned geographical data

Payment, review and post-delivery information are excluded from predictive features where they could introduce target leakage or use information unavailable when freight cost is estimated.

## Exploratory and Statistical Analysis

The project performs extensive statistical analysis before modelling, including:

- measures of central tendency
- measures of spread
- distribution analysis
- skewness
- excess kurtosis
- IQR-based outlier analysis
- modified-z diagnostics
- Pearson correlation
- Spearman correlation
- categorical-cardinality analysis
- Q-Q analysis

The 9,000-row analytical sample contains a strongly right-skewed freight-cost distribution.

Recorded target statistics include:

```text
Mean freight value:       19.70 BRL
Median freight value:     16.25 BRL
95th percentile:          44.02 BRL
99th percentile:          82.583 BRL
Maximum freight value:   306.06 BRL
Skewness:                  4.958
Excess kurtosis:          44.644
```

## Leakage-Safe Preprocessing

Model preprocessing is learned from training observations rather than from the complete dataset.

The workflow includes:

- numerical median imputation
- feature scaling
- categorical imputation
- one-hot encoding
- rare-category handling
- group-aware train/test splitting
- group-aware cross-validation

Items belonging to the same order are kept within the same split so related observations do not leak between training and evaluation data.

## Feature Engineering

Domain-informed variables are introduced to represent physical and geographical factors affecting freight cost.

Examples include:

- parcel volume
- parcel density
- seller-to-customer distance
- price-to-weight relationships
- logarithmic transformations
- squared distance
- squared volume
- weight × distance interactions
- volume × distance interactions
- same-state delivery context

## Regression Models

Three main regression approaches are evaluated.

### Baseline Multiple Linear Regression

Reported test performance:

```text
RMSE: 11.492 BRL
MAE:   5.166 BRL
R²:    0.562
```

### Feature-Engineered Ordinary Least Squares

Reported test performance:

```text
RMSE: 10.601 BRL
MAE:   4.755 BRL
R²:    0.627
```

### Tuned Ridge Regression

Reported test performance:

```text
RMSE: 10.629 BRL
MAE:   4.740 BRL
R²:    0.625
```

The feature-engineered OLS model achieves the lowest single-split RMSE, while the tuned Ridge model achieves the lowest MAE and provides greater coefficient stability when correlated engineered features are present.

The project selects **Ridge regression** as the preferred final model after considering:

- test error
- median absolute error
- group-aware cross-validation
- regularisation
- coefficient stability

Relative to the baseline model, Ridge reduces:

```text
RMSE by approximately 7.5%
MAE  by approximately 8.3%
```

## Regression Diagnostics

The modelling stage includes detailed statistical diagnostics rather than relying only on predictive scores.

The analysis includes:

- residual-vs-fitted behaviour
- actual-vs-predicted calibration
- Breusch-Pagan heteroscedasticity testing
- variance inflation factors
- leverage
- Cook's distance
- studentised residuals
- influential-observation analysis

## Robustness Analysis

Final-model uncertainty is evaluated using **order-level bootstrap resampling**.

Resampling by order preserves dependence between multiple items belonging to the same order.

The reported analysis found that Ridge reduced RMSE relative to the baseline in **96.4% of order-level bootstrap samples** and reduced MAE in every bootstrap sample.

Performance is weaker for unusually expensive deliveries.

The highest freight-cost quartile has a reported:

```text
RMSE: 19.447 BRL
```

The model is therefore interpreted as a freight-cost estimation and planning tool rather than a fully automated pricing system.

## Main Findings

The analysis indicates that freight cost is strongly associated with combinations of:

- parcel volume
- product weight
- geographical distance
- interactions between physical parcel characteristics and distance
- regional delivery context

The regression coefficients are interpreted as statistical associations rather than causal effects.

## Repository Structure

```text
olist-freight-cost-regression/
├── README.md
├── requirements.txt
├── .gitignore
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
    └── original_notebook_export.html
```

## Original Project Files

The original supplied project artefacts are retained without rewriting the implementation:

- `docs/original_notebook_export.html` — original Jupyter Notebook HTML export containing the project code, narrative and outputs
- `docs/original_coursework_report.pdf` — original PDF export of the completed project

## Dataset Setup

The raw Olist CSV files are not included in the repository.

The file:

```text
data/raw/README.md
```

lists the exact dataset filenames used by the project.

## Main Technologies

- Python
- pandas
- NumPy
- SciPy
- Matplotlib
- scikit-learn
- statsmodels
- Jupyter Notebook

## Techniques Used

**Data preparation:** missing-value analysis, duplicate handling, type conversion, sanitisation and plausibility checks

**Relational data:** candidate and foreign keys, multiplicity analysis, controlled joins and analytical-table construction

**Feature engineering:** Haversine distance, parcel metrics, transformations and interaction variables

**Statistical analysis:** distributions, correlations, outliers, heteroscedasticity, multicollinearity and influence diagnostics

**Machine learning:** multiple linear regression, Ridge regression, group-aware cross-validation and leakage-safe preprocessing

**Evaluation:** RMSE, MAE, R², residual analysis, order-level bootstrap and subgroup error analysis

## Author

**Zimmel Javed Virk**  
BSc Computer Science — University of London
