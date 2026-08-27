"""
Mini Project: Sales Data Wrangling, Preprocessing & Customer Segmentation
==========================================================================
Stack: pandas, numpy, matplotlib, scikit-learn

Pipeline:
1. Generate a realistic "messy" retail sales dataset (missing values,
   duplicates, inconsistent text, outliers) - mimics a raw export.
2. Clean & preprocess it with pandas/numpy.
3. Engineer features (profit margin, order month, RFM-style metrics).
4. Visualize patterns with matplotlib.
5. Use scikit-learn (StandardScaler + KMeans) to segment customers,
   then visualize the segments.

Run: python sales_insights_project.py
Outputs land in ./outputs/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import os

np.random.seed(42)
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# 1. GENERATE A MESSY RAW DATASET (simulates a real-world dirty export)
# --------------------------------------------------------------------------
N = 2000
customers = [f"CUST-{i:04d}" for i in range(1, 301)]
categories = ["Furniture", "Office Supplies", "Technology", "furniture", "TECH"]  # inconsistent casing on purpose
regions = ["East", "West", "North", "South", None]

dates = pd.date_range("2023-01-01", "2024-12-31", periods=N)

raw = pd.DataFrame({
    "Order_ID": [f"ORD-{i}" for i in range(N)],
    "Order_Date": np.random.choice(dates, N),
    "Customer_ID": np.random.choice(customers, N),
    "Category": np.random.choice(categories, N, p=[0.25, 0.35, 0.2, 0.1, 0.1]),
    "Region": np.random.choice(regions, N, p=[0.27, 0.27, 0.23, 0.2, 0.03]),
    "Sales": np.round(np.random.gamma(shape=2.0, scale=120, size=N), 2),
    "Quantity": np.random.randint(1, 15, N),
    "Discount": np.round(np.random.choice([0, 0.1, 0.2, 0.3, np.nan], N, p=[0.4, 0.25, 0.15, 0.1, 0.1]), 2),
    "Profit": np.round(np.random.normal(30, 60, N), 2),
})

# inject duplicates
raw = pd.concat([raw, raw.sample(40, random_state=1)], ignore_index=True)

# inject some missing Sales / Customer_ID
raw.loc[raw.sample(frac=0.03, random_state=2).index, "Sales"] = np.nan
raw.loc[raw.sample(frac=0.02, random_state=3).index, "Customer_ID"] = None

# inject a few extreme outliers in Sales
outlier_idx = raw.sample(6, random_state=4).index
raw.loc[outlier_idx, "Sales"] = raw.loc[outlier_idx, "Sales"] * 25

raw.to_csv(f"{OUT_DIR}/raw_sales_data.csv", index=False)
print(f"Raw dataset shape: {raw.shape}")
print(raw.isna().sum())

# --------------------------------------------------------------------------
# 2. CLEANING & PREPROCESSING
# --------------------------------------------------------------------------
df = raw.copy()

# 2a. Drop exact duplicate rows
before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} duplicate rows")

# 2b. Standardize text fields
df["Category"] = df["Category"].str.strip().str.title().replace({"Tech": "Technology"})
df["Region"] = df["Region"].fillna("Unknown")

# 2c. Handle missing Customer_ID -> drop (can't segment an unknown customer)
df = df.dropna(subset=["Customer_ID"])

# 2d. Handle missing Sales -> impute with category median (more robust than global mean)
df["Sales"] = df.groupby("Category")["Sales"].transform(lambda s: s.fillna(s.median()))

# 2e. Handle missing Discount -> assume no discount recorded means 0
df["Discount"] = df["Discount"].fillna(0)

# 2f. Outlier treatment on Sales using IQR capping (winsorizing), not deletion,
#     to preserve row count for downstream aggregation
q1, q3 = df["Sales"].quantile([0.25, 0.75])
iqr = q3 - q1
upper_cap = q3 + 1.5 * iqr
n_capped = (df["Sales"] > upper_cap).sum()
df["Sales"] = np.where(df["Sales"] > upper_cap, upper_cap, df["Sales"])
print(f"Capped {n_capped} outlier rows in Sales at {upper_cap:.2f}")

# 2g. Correct dtypes
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Quantity"] = df["Quantity"].astype(int)

# --------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# --------------------------------------------------------------------------
df["Order_Month"] = df["Order_Date"].dt.to_period("M").astype(str)
df["Order_Weekday"] = df["Order_Date"].dt.day_name()
df["Net_Sales"] = df["Sales"] * (1 - df["Discount"])
df["Profit_Margin"] = df["Profit"] / df["Net_Sales"].replace(0, np.nan)

df.to_csv(f"{OUT_DIR}/cleaned_sales_data.csv", index=False)
print(f"\nCleaned dataset shape: {df.shape}")

# --------------------------------------------------------------------------
# 4. VISUALIZATION (matplotlib)
# --------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# 4a. Monthly sales trend
monthly = df.groupby("Order_Month")["Net_Sales"].sum().sort_index()
axes[0, 0].plot(monthly.index, monthly.values, marker="o", color="#2563eb")
axes[0, 0].set_title("Monthly Net Sales Trend")
axes[0, 0].tick_params(axis="x", rotation=75, labelsize=7)
axes[0, 0].set_ylabel("Net Sales ($)")

# 4b. Sales by category
cat_sales = df.groupby("Category")["Net_Sales"].sum().sort_values(ascending=False)
axes[0, 1].bar(cat_sales.index, cat_sales.values, color="#16a34a")
axes[0, 1].set_title("Total Net Sales by Category")
axes[0, 1].set_ylabel("Net Sales ($)")

# 4c. Sales distribution before/after outlier capping
axes[1, 0].boxplot([raw["Sales"].dropna(), df["Sales"]], tick_labels=["Raw", "Cleaned"])
axes[1, 0].set_title("Sales Distribution: Raw vs Cleaned (Outlier Capping)")
axes[1, 0].set_ylabel("Sales ($)")

# 4d. Sales by region
region_sales = df.groupby("Region")["Net_Sales"].sum().sort_values(ascending=False)
axes[1, 1].bar(region_sales.index, region_sales.values, color="#f59e0b")
axes[1, 1].set_title("Total Net Sales by Region")
axes[1, 1].set_ylabel("Net Sales ($)")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_overview.png", dpi=150)
plt.close()
print("Saved eda_overview.png")

# --------------------------------------------------------------------------
# 5. SCIKIT-LEARN: RFM-STYLE CUSTOMER SEGMENTATION (KMeans)
# --------------------------------------------------------------------------
snapshot_date = df["Order_Date"].max() + pd.Timedelta(days=1)

rfm = df.groupby("Customer_ID").agg(
    Recency=("Order_Date", lambda x: (snapshot_date - x.max()).days),
    Frequency=("Order_ID", "count"),
    Monetary=("Net_Sales", "sum"),
).reset_index()

# Scale features so KMeans isn't dominated by Monetary's larger range
features = rfm[["Recency", "Frequency", "Monetary"]]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["Segment"] = kmeans.fit_predict(scaled)

# Label segments by average Monetary value for readability
seg_order = rfm.groupby("Segment")["Monetary"].mean().sort_values(ascending=False).index
label_map = {seg: name for seg, name in zip(seg_order, ["Champions", "Loyal", "At Risk", "Low Value"])}
rfm["Segment_Label"] = rfm["Segment"].map(label_map)

rfm.to_csv(f"{OUT_DIR}/customer_segments.csv", index=False)
print("\nCustomer segment sizes:")
print(rfm["Segment_Label"].value_counts())

# 5a. Visualize segments via PCA (reduce 3 features to 2D for plotting)
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(scaled)
rfm["PC1"], rfm["PC2"] = coords[:, 0], coords[:, 1]

fig, ax = plt.subplots(figsize=(8, 6))
colors = {"Champions": "#dc2626", "Loyal": "#2563eb", "At Risk": "#f59e0b", "Low Value": "#6b7280"}
for label, group in rfm.groupby("Segment_Label"):
    ax.scatter(group["PC1"], group["PC2"], label=label, alpha=0.7, s=40, color=colors.get(label))
ax.set_title("Customer Segments (KMeans, PCA-reduced RFM features)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/customer_segments.png", dpi=150)
plt.close()
print("Saved customer_segments.png")

# 5b. Segment profile bar chart
profile = rfm.groupby("Segment_Label")[["Recency", "Frequency", "Monetary"]].mean()
profile_norm = (profile - profile.min()) / (profile.max() - profile.min())  # normalize for comparability

fig, ax = plt.subplots(figsize=(8, 5))
profile_norm.plot(kind="bar", ax=ax, color=["#2563eb", "#16a34a", "#f59e0b"])
ax.set_title("Normalized Segment Profiles (Recency, Frequency, Monetary)")
ax.set_ylabel("Normalized Score (0-1)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/segment_profiles.png", dpi=150)
plt.close()
print("Saved segment_profiles.png")

print("\nDone. All outputs saved to ./outputs/")
