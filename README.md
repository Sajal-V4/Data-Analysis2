# Sales Data Wrangling, Preprocessing & Customer Segmentation

A small end-to-end project demonstrating data cleaning, feature engineering,
visualization, and machine learning on retail sales data.

**Stack:** `pandas`, `numpy`, `matplotlib`, `scikit-learn`

## What it does

1. **Generates a messy raw dataset** (`raw_sales_data.csv`) — synthetic retail
   sales data with the problems real-world exports have: duplicate rows,
   missing values, inconsistent category casing (`"Tech"` vs `"Technology"`),
   and injected outliers.

2. **Cleans & preprocesses it** with pandas/numpy:
   - Drops duplicate rows
   - Standardizes inconsistent text fields
   - Imputes missing `Sales` using category-wise median
   - Fills missing `Discount` with 0
   - Caps outliers in `Sales` using the IQR method (winsorizing, not deletion)
   - Fixes data types (dates, integers)

3. **Engineers features**: order month, order weekday, net sales
   (after discount), profit margin.

4. **Visualizes** (`eda_overview.png`) — a 4-panel figure:
   - Monthly net sales trend
   - Total sales by category
   - Raw vs. cleaned sales distribution (shows the outlier fix)
   - Total sales by region

5. **Applies scikit-learn** for customer segmentation:
   - Builds RFM (Recency, Frequency, Monetary) features per customer
   - Scales them with `StandardScaler`
   - Runs `KMeans` (k=4) to segment customers into Champions / Loyal /
     At Risk / Low Value
   - Uses `PCA` to reduce the 3 RFM dimensions to 2D for plotting
     (`customer_segments.png`)
   - Compares segment profiles in a normalized bar chart
     (`segment_profiles.png`)

## Files

| File | Description |
|---|---|
| `sales_insights_project.py` | The full pipeline script — run it to regenerate everything |
| `raw_sales_data.csv` | Synthetic messy dataset before cleaning |
| `cleaned_sales_data.csv` | Dataset after cleaning + feature engineering |
| `customer_segments.csv` | Per-customer RFM values and assigned segment |
| `eda_overview.png` | 4-panel exploratory data analysis chart |
| `customer_segments.png` | PCA scatter plot of the 4 customer segments |
| `segment_profiles.png` | Normalized Recency/Frequency/Monetary comparison across segments |

## How to run

```bash
pip install pandas numpy matplotlib scikit-learn
python sales_insights_project.py
```

Outputs are written to an `outputs/` folder next to the script.

## Notes / ideas for extending it

- Swap the synthetic data generator for a real dataset (e.g. a Superstore-style
  export) — the cleaning → feature engineering → visualization → segmentation
  pipeline carries over largely unchanged.
- Try a different `k` for KMeans and use the elbow method or silhouette score
  to justify the choice.
- Add a regression model (e.g. `LinearRegression` or `RandomForestRegressor`)
  to predict `Net_Sales` or `Profit_Margin` from category/region/discount.
