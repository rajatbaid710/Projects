# =============================================================================
# Exploratory Data Analysis (EDA) — Nutritional Status Classification
# Output: eda_plots/ folder + eda_report.html
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os

from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION — Edit these before running
# =============================================================================

EXCEL_FILE     = "/Users/rajatbaid/Documents/data/NutritionDataSet.xlsx"       # Path to your Excel file
TARGET_COLUMN  = "Nutrition_Route_ESPEN"     # Column name with nutrition labels
COLUMNS_TO_DROP = []                    # e.g., ["patient_id", "name", "date"]
OUTPUT_DIR     = "/Users/rajatbaid/Documents/code/Projects/data/eda_plots"            # Folder to save all plots

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plotting style
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.05)
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

COLORS = sns.color_palette("Set2")

# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)

df = pd.read_excel(EXCEL_FILE)

if COLUMNS_TO_DROP:
    df.drop(columns=COLUMNS_TO_DROP, inplace=True)

print(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
print(f"\nColumn types:\n{df.dtypes.value_counts()}")

numeric_cols     = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

if TARGET_COLUMN in numeric_cols:
    numeric_cols.remove(TARGET_COLUMN)
if TARGET_COLUMN in categorical_cols:
    categorical_cols.remove(TARGET_COLUMN)

print(f"\nNumeric features   : {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")
print(f"Target classes     : {df[TARGET_COLUMN].unique()}")

# =============================================================================
# 2. DATA QUALITY OVERVIEW
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Data Quality")
print("=" * 60)

# Summary stats
print("\nDescriptive Statistics:")
print(df[numeric_cols].describe().round(2).to_string())

# Missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing": missing, "% Missing": missing_pct})
missing_df = missing_df[missing_df["Missing"] > 0].sort_values("% Missing", ascending=False)

if not missing_df.empty:
    print(f"\nColumns with missing values:\n{missing_df.to_string()}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Missing Data Analysis", fontsize=14, fontweight="bold")

    # Bar chart of missing %
    missing_df["% Missing"].plot(kind="bar", ax=axes[0], color=COLORS[0], edgecolor="white")
    axes[0].set_title("% Missing per Column")
    axes[0].set_ylabel("% Missing")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")

    # Heatmap
    sns.heatmap(df[missing_df.index].isnull(), cbar=False, yticklabels=False,
                cmap="Blues", ax=axes[1])
    axes[1].set_title("Missing Value Pattern\n(blue = missing)")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_missing_data.png")
    plt.close()
    print(f"  → Saved: {OUTPUT_DIR}/01_missing_data.png")
else:
    print("  No missing values found.")

# =============================================================================
# 3. CLASS DISTRIBUTION
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Class Distribution")
print("=" * 60)

class_counts = df[TARGET_COLUMN].value_counts()
class_pct    = (class_counts / len(df) * 100).round(1)

print(f"\nClass counts:\n{class_counts.to_string()}")
print(f"\nClass percentages:\n{class_pct.to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Nutritional Status Distribution", fontsize=14, fontweight="bold")

# Bar chart
axes[0].bar(class_counts.index, class_counts.values,
            color=COLORS[:len(class_counts)], edgecolor="white")
axes[0].set_title("Sample Count per Class")
axes[0].set_ylabel("Count")
axes[0].set_xticklabels(class_counts.index, rotation=30, ha="right")
for i, (count, pct) in enumerate(zip(class_counts.values, class_pct.values)):
    axes[0].text(i, count + 0.5, f"{pct}%", ha="center", fontsize=10)

# Pie chart
axes[1].pie(class_counts.values, labels=class_counts.index, autopct="%1.1f%%",
            colors=COLORS[:len(class_counts)], startangle=90)
axes[1].set_title("Proportion per Class")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_class_distribution.png")
plt.close()
print(f"  → Saved: {OUTPUT_DIR}/02_class_distribution.png")

# =============================================================================
# 4. UNIVARIATE — DISTRIBUTIONS OF NUMERIC FEATURES
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Univariate Analysis (Numeric Features)")
print("=" * 60)

n_cols = 3
n_rows = int(np.ceil(len(numeric_cols) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
fig.suptitle("Distribution of Numeric Features", fontsize=14, fontweight="bold", y=1.01)
axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    data = df[col].dropna()
    ax.hist(data, bins=25, color=COLORS[i % len(COLORS)], edgecolor="white", alpha=0.85)
    ax.axvline(data.mean(),   color="red",    linestyle="--", linewidth=1.2, label="Mean")
    ax.axvline(data.median(), color="orange", linestyle=":",  linewidth=1.2, label="Median")
    ax.set_title(col, fontsize=11)
    ax.set_xlabel("")
    ax.legend(fontsize=8)
    skew = round(data.skew(), 2)
    ax.text(0.98, 0.92, f"Skew: {skew}", transform=ax.transAxes,
            ha="right", fontsize=8.5, color="gray")

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_numeric_distributions.png")
plt.close()
print(f"  → Saved: {OUTPUT_DIR}/03_numeric_distributions.png")

# =============================================================================
# 5. BOXPLOTS BY NUTRITION CLASS
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Group Comparison by Nutrition Status")
print("=" * 60)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
fig.suptitle("Feature Distributions by Nutrition Status", fontsize=14, fontweight="bold", y=1.01)
axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    groups = [df[df[TARGET_COLUMN] == cls][col].dropna().values
              for cls in df[TARGET_COLUMN].unique()]
    bp = ax.boxplot(groups, patch_artist=True,
                    labels=df[TARGET_COLUMN].unique(),
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_title(col, fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_boxplots_by_class.png")
plt.close()
print(f"  → Saved: {OUTPUT_DIR}/04_boxplots_by_class.png")

# =============================================================================
# 6. OUTLIER DETECTION
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Outlier Detection (IQR method)")
print("=" * 60)

outlier_summary = {}
for col in numeric_cols:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    n_outliers = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    outlier_summary[col] = n_outliers

outlier_df = pd.Series(outlier_summary).sort_values(ascending=False)
print(f"\nOutlier counts (IQR method):\n{outlier_df[outlier_df > 0].to_string()}")

fig, ax = plt.subplots(figsize=(10, 4))
outlier_df[outlier_df > 0].plot(kind="bar", ax=ax, color=COLORS[3], edgecolor="white")
ax.set_title("Outlier Count per Feature (IQR Method)", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Outliers")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_outliers.png")
plt.close()
print(f"  → Saved: {OUTPUT_DIR}/05_outliers.png")

# =============================================================================
# 7. CORRELATION HEATMAP
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: Correlation Heatmap")
print("=" * 60)

corr_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols)), max(6, len(numeric_cols) * 0.8)))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, linewidths=0.5,
            annot_kws={"size": 9}, ax=ax)
ax.set_title("Feature Correlation Matrix (Pearson)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_correlation_heatmap.png")
plt.close()
print(f"  → Saved: {OUTPUT_DIR}/06_correlation_heatmap.png")

# Highlight highly correlated pairs
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j],
                              round(corr_matrix.iloc[i, j], 3)))
if high_corr:
    print(f"\nHighly correlated pairs (|r| > 0.7):")
    for a, b, r in high_corr:
        print(f"  {a} ↔ {b}: r = {r}")

# =============================================================================
# 8. STATISTICAL SIGNIFICANCE (ANOVA)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8: ANOVA — Feature Significance per Class")
print("=" * 60)

anova_results = {}
for col in numeric_cols:
    groups = [df[df[TARGET_COLUMN] == cls][col].dropna().values
              for cls in df[TARGET_COLUMN].unique()]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) >= 2:
        f_stat, p_val = stats.f_oneway(*groups)
        anova_results[col] = {"F-statistic": round(f_stat, 3), "p-value": round(p_val, 4)}

anova_df = pd.DataFrame(anova_results).T.sort_values("p-value")
print(f"\nANOVA results (sorted by p-value):\n{anova_df.to_string()}")

fig, ax = plt.subplots(figsize=(10, 5))
colors_bar = [COLORS[0] if p < 0.05 else COLORS[3] for p in anova_df["p-value"]]
bars = ax.barh(anova_df.index, -np.log10(anova_df["p-value"]),
               color=colors_bar, edgecolor="white")
ax.axvline(-np.log10(0.05), color="red", linestyle="--", linewidth=1.2,
           label="p = 0.05 threshold")
ax.set_xlabel("-log10(p-value)")
ax.set_title("Feature Significance Across Nutrition Classes (ANOVA)", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_anova_significance.png")
plt.close()
print(f"  → Saved: {OUTPUT_DIR}/07_anova_significance.png")

# =============================================================================
# 9. PCA — 2D VISUALIZATION
# =============================================================================

print("\n" + "=" * 60)
print("STEP 9: PCA — 2D Visualization")
print("=" * 60)

df_pca = df[numeric_cols].fillna(df[numeric_cols].median())
X_scaled = StandardScaler().fit_transform(df_pca)

pca = PCA(n_components=2, random_state=42)
components = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_ * 100

le = LabelEncoder()
labels = le.fit_transform(df[TARGET_COLUMN])
class_names = le.classes_

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("PCA — 2D Projection", fontsize=14, fontweight="bold")

# Scatter by class
for i, cls in enumerate(class_names):
    mask = labels == i
    axes[0].scatter(components[mask, 0], components[mask, 1],
                    c=[COLORS[i]], label=cls, alpha=0.75, s=50, edgecolors="white")
axes[0].set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
axes[0].set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
axes[0].set_title("Samples colored by Nutrition Status")
axes[0].legend(title="Status", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)

# Scree plot (full PCA)
pca_full = PCA(random_state=42).fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_ * 100)
axes[1].bar(range(1, len(cumvar) + 1), pca_full.explained_variance_ratio_ * 100,
            color=COLORS[0], alpha=0.7, label="Individual")
axes[1].plot(range(1, len(cumvar) + 1), cumvar, "o-", color=COLORS[1],
             linewidth=1.5, markersize=4, label="Cumulative")
axes[1].axhline(80, color="red", linestyle="--", linewidth=1, label="80% threshold")
axes[1].set_xlabel("Principal Component")
axes[1].set_ylabel("Explained Variance (%)")
axes[1].set_title("Scree Plot")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_pca.png")
plt.close()
print(f"  → Saved: {OUTPUT_DIR}/08_pca.png")

# =============================================================================
# 10. PRE vs POST SMOTE CLASS BALANCE
# =============================================================================

print("\n" + "=" * 60)
print("STEP 10: Pre vs Post SMOTE Comparison")
print("=" * 60)

X_num = df[numeric_cols].fillna(df[numeric_cols].median())
y_num = le.transform(df[TARGET_COLUMN])

smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_num, y_num)

before = pd.Series(y_num).value_counts().sort_index()
after  = pd.Series(y_sm).value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Class Balance: Before vs After SMOTE", fontsize=14, fontweight="bold")

axes[0].bar(le.classes_, before.values,
            color=COLORS[:len(le.classes_)], edgecolor="white")
axes[0].set_title(f"Before SMOTE  (n={len(y_num)})")
axes[0].set_ylabel("Count")
for i, v in enumerate(before.values):
    axes[0].text(i, v + 0.3, str(v), ha="center", fontsize=10)

axes[1].bar(le.classes_, after.values,
            color=COLORS[:len(le.classes_)], edgecolor="white")
axes[1].set_title(f"After SMOTE  (n={len(y_sm)})")
for i, v in enumerate(after.values):
    axes[1].text(i, v + 0.3, str(v), ha="center", fontsize=10)

for ax in axes:
    ax.set_xticklabels(le.classes_, rotation=30, ha="right")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_smote_comparison.png")
plt.close()
print(f"  → Saved: {OUTPUT_DIR}/09_smote_comparison.png")

# =============================================================================
# 11. AUTOMATED EDA REPORT (ydata-profiling)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 11: Automated EDA Report (ydata-profiling)")
print("=" * 60)

try:
    from ydata_profiling import ProfileReport

    profile = ProfileReport(
        df,
        title="Nutritional Status — EDA Report",
        explorative=True,
        correlations={"pearson": {"calculate": True}, "spearman": {"calculate": True}},
        missing_diagrams={"heatmap": True, "bar": True},
    )
    report_path = f"{OUTPUT_DIR}/eda_report.html"
    profile.to_file(report_path)
    print(f"  → Full HTML report saved: {report_path}")

except ImportError:
    print("  ydata-profiling not installed.")
    print("  Install with: pip install ydata-profiling")
    print("  Then re-run this step for a complete interactive HTML report.")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("EDA COMPLETE — Outputs")
print("=" * 60)

files = sorted(os.listdir(OUTPUT_DIR))
for f in files:
    print(f"  {OUTPUT_DIR}/{f}")

print(f"""
Key findings to check:
  1. Class imbalance  → {OUTPUT_DIR}/02_class_distribution.png
  2. Skewed features  → {OUTPUT_DIR}/03_numeric_distributions.png
  3. Class separation → {OUTPUT_DIR}/04_boxplots_by_class.png
  4. Outliers         → {OUTPUT_DIR}/05_outliers.png
  5. Correlated pairs → {OUTPUT_DIR}/06_correlation_heatmap.png
  6. Most significant → {OUTPUT_DIR}/07_anova_significance.png
  7. 2D cluster view  → {OUTPUT_DIR}/08_pca.png
  8. SMOTE effect     → {OUTPUT_DIR}/09_smote_comparison.png
""")
