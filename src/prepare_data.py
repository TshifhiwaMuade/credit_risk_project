from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Project paths (portable)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "credit_risk_dataset.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

# Clear old PNG files before generating new ones
for file in FIGURES_DIR.glob("*.png"):
    file.unlink()


# 2. Check raw file exists
if not RAW_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Could not find dataset at: {RAW_DATA_PATH}\n"
        f"Make sure 'credit_risk_dataset.csv' is inside data/raw/"
    )


# 3. Load dataset
df = pd.read_csv(RAW_DATA_PATH)

# 4. Check and remove duplicates
duplicate_count = int(df.duplicated().sum())

print("\n=== DUPLICATE ROWS BEFORE CLEANING ===")
print(duplicate_count)

df = df.drop_duplicates().copy()

print("\n=== SHAPE AFTER DROPPING DUPLICATES ===")
print(df.shape)


# 5. Dataset overview
print("\n=== DATASET OVERVIEW ===")
print("Shape:", df.shape)

print("\n=== COLUMNS ===")
print(df.columns.tolist())

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== MISSING VALUES BEFORE CLEANING ===")
print(df.isnull().sum())

print("\n=== SUMMARY STATISTICS ===")
print(df.describe(include="all"))

print("\n=== CLASS DISTRIBUTION ===")
print(df["loan_status"].value_counts())
print(df["loan_status"].value_counts(normalize=True).mul(100).round(2))

# 6. Plot class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="loan_status", data=df)
plt.title("Class Distribution of loan_status")
plt.xlabel("loan_status (0 = No Default, 1 = Default)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "class_distribution.png")
plt.close()


# 7. Plot numeric distributions
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if "loan_status" in numeric_cols:
    numeric_cols.remove("loan_status")

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col].dropna(), kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{col}_distribution.png")
    plt.close()


# 8. Plot categorical distributions
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    order = df[col].value_counts().index
    sns.countplot(x=col, data=df, order=order)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{col}_distribution.png")
    plt.close()


# 9. Correlation heatmap
plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "correlation_heatmap.png")
plt.close()


# 10. Boxplots before cleaning
outlier_check_cols = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

for col in outlier_check_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col} (Before Cleaning)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{col}_boxplot_before.png")
    plt.close()


# 11. IQR outlier summary function
def iqr_outlier_summary(dataframe, column_name):
    series = dataframe[column_name].dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = series[(series < lower_bound) | (series > upper_bound)]

    return {
        "column": column_name,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": int(len(outliers)),
    }


# 12. Outlier checks before cleaning
print("\n=== IQR OUTLIER SUMMARY BEFORE CLEANING ===")
iqr_columns = ["person_income", "loan_amnt", "person_age", "person_emp_length"]

for col in iqr_columns:
    summary = iqr_outlier_summary(df, col)
    print(f"\nColumn: {summary['column']}")
    print(f"Q1: {summary['q1']}")
    print(f"Q3: {summary['q3']}")
    print(f"IQR: {summary['iqr']}")
    print(f"Lower bound: {summary['lower_bound']}")
    print(f"Upper bound: {summary['upper_bound']}")
    print(f"Outlier count: {summary['outlier_count']}")


# 13. Data cleaning
# Fill person_emp_length with median
df["person_emp_length"] = df["person_emp_length"].fillna(
    df["person_emp_length"].median()
)

# Fill loan_int_rate with median grouped by loan_intent
df["loan_int_rate"] = df.groupby("loan_intent")["loan_int_rate"].transform(
    lambda x: x.fillna(x.median())
)

# Fallback in case any values remain missing
df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())

# Cap unrealistic ages above 70
df["person_age"] = df["person_age"].clip(upper=70)


# 14. Checks after cleaning
print("\n=== MISSING VALUES AFTER CLEANING ===")
print(df.isnull().sum())

print("\n=== MAX AGE AFTER CAPPING ===")
print(df["person_age"].max())

print("\n=== FINAL SHAPE AFTER CLEANING ===")
print(df.shape)

print("\n=== IQR OUTLIER SUMMARY AFTER CLEANING ===")
for col in iqr_columns:
    summary = iqr_outlier_summary(df, col)
    print(f"\nColumn: {summary['column']}")
    print(f"Q1: {summary['q1']}")
    print(f"Q3: {summary['q3']}")
    print(f"IQR: {summary['iqr']}")
    print(f"Lower bound: {summary['lower_bound']}")
    print(f"Upper bound: {summary['upper_bound']}")
    print(f"Outlier count: {summary['outlier_count']}")


# 15. Boxplots after cleaning
for col in outlier_check_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col} (After Cleaning)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{col}_boxplot_after.png")
    plt.close()


# 16. Save cleaned dataset
df.to_csv(PROCESSED_DATA_PATH, index=False)

print(f"\nCleaned dataset saved to:\n{PROCESSED_DATA_PATH}")
print(f"Figures saved to:\n{FIGURES_DIR}")