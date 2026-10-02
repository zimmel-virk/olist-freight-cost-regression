# Portfolio notes

This GitHub version is derived from the user's completed CM3005 Data Science coursework.

## Preserved

- project question and Olist dataset;
- nine-file relational workflow;
- data-quality auditing;
- key and relationship-integrity analysis;
- cleaning and sanitisation logic;
- Haversine distance calculation;
- group-aware train/test and cross-validation strategy;
- baseline linear regression;
- feature-engineered OLS;
- tuned Ridge regression;
- diagnostic and bootstrap methodology;
- reported model metrics and interpretation.

## Repository refactor

The original Jupyter HTML export was converted into a clean `.ipynb` file so GitHub can display the notebook directly. Cell outputs were omitted from the reconstructed notebook because the original HTML and PDF already preserve the completed outputs and because excluding embedded outputs keeps the Git repository significantly smaller.

A small `src/freight_utils.py` module exposes reusable functions from the notebook without changing the analytical methodology.

The raw third-party Olist CSV files are not bundled.
