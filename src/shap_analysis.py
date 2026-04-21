from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
DATA_DIR = BASE_DIR / "data" / "processed"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load Models and Data ───────────────────────────────────────────────────────
print("Loading models and data...")

# Load best classifier (XGBoost)
xgb_model = joblib.load(ARTIFACTS_DIR / "model_xgb.pkl")

# Load preprocessor from Stephen
preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.pkl")

# Load K-Means model and scaler from Nathan
kmeans_model = joblib.load(ARTIFACTS_DIR / "kmeans_model.pkl")
cluster_scaler = joblib.load(ARTIFACTS_DIR / "cluster_scaler.pkl")

# Load cluster labels and profiles
cluster_labels = pd.read_csv(ARTIFACTS_DIR / "cluster_labels.csv")
cluster_profiles = pd.read_csv(ARTIFACTS_DIR / "cluster_profiles.csv")

# Load test data
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

print(f"X_test shape: {X_test.shape}")
print(f"Cluster labels shape: {cluster_labels.shape}")
print(f"Cluster profiles:\n{cluster_profiles}")

# ── Feature Names ──────────────────────────────────────────────────────────────
# Get feature names from preprocessor
feature_names = preprocessor.get_feature_names_out().tolist()

# Clean up feature names for better readability in plots
clean_names = [
    name.replace("num__", "").replace("cat__", "").replace("_", " ")
    for name in feature_names
]

print(f"\nFeature names: {clean_names}")

# ── SECTION 1 — SHAP on XGBoost Classifier ────────────────────────────────────
print("\n" + "="*60)
print("SECTION 1: SHAP Analysis on XGBoost Classifier")
print("="*60)

# Use TreeExplainer for XGBoost — fastest and most accurate for tree models
explainer_xgb = shap.TreeExplainer(xgb_model)

# Use a sample of 2000 rows to keep computation manageable
sample_size = 2000
X_test_sample = X_test.iloc[:sample_size]
y_test_sample = y_test.iloc[:sample_size]

print(f"\nComputing SHAP values on {sample_size} test samples...")
shap_values_xgb = explainer_xgb.shap_values(X_test_sample)

print(f"SHAP values shape: {np.array(shap_values_xgb).shape}")

# ── Global Feature Importance — Bar Plot ───────────────────────────────────────
print("\nGenerating global bar plot...")

# Convert to DataFrame with clean names for plotting
X_test_named = pd.DataFrame(X_test_sample.values, columns=clean_names)

plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values_xgb,
    X_test_named,
    plot_type="bar",
    show=False
)
plt.title("Global Feature Importance — XGBoost Credit Risk (Bar)", fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shap_bar.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved shap_bar.png")

# ── Global Feature Importance — Beeswarm Plot ─────────────────────────────────
print("\nGenerating beeswarm plot...")

plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values_xgb,
    X_test_named,
    show=False
)
plt.title("Global Feature Importance — XGBoost Credit Risk (Beeswarm)", fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shap_beeswarm.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved shap_beeswarm.png")

# ── Local Explanation — Waterfall Plot ────────────────────────────────────────
print("\nGenerating waterfall plot for a high-risk applicant...")

# Find a high-risk applicant (predicted default = 1) for a more interesting explanation
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_test_enc = le.fit_transform(y_test_sample)
xgb_preds = xgb_model.predict(X_test_sample)

# Find first predicted default applicant
default_indices = np.where(xgb_preds == 1)[0]
patient_idx = int(default_indices[0]) if len(default_indices) > 0 else 0
print(f"Explaining applicant at index {patient_idx} (predicted default)")

# Handle both binary and multiclass SHAP output
shap_arr = np.array(shap_values_xgb)

if shap_arr.ndim == 3:
    # Multiclass — use class 1 (Default)
    sv = shap_arr[patient_idx, :, 1]
    base_val = explainer_xgb.expected_value[1] if hasattr(explainer_xgb.expected_value, '__len__') else explainer_xgb.expected_value
else:
    sv = shap_arr[patient_idx, :]
    base_val = explainer_xgb.expected_value

explanation = shap.Explanation(
    values=sv,
    base_values=base_val,
    data=X_test_named.iloc[patient_idx].values,
    feature_names=clean_names
)

plt.figure()
shap.plots.waterfall(explanation, show=False)
plt.title(f"Local Explanation — High-Risk Applicant (Default Class)", fontsize=11, pad=15)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shap_waterfall.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved shap_waterfall.png")

# ── Save SHAP Values to CSV ───────────────────────────────────────────────────
print("\nSaving SHAP values to CSV...")

shap_arr = np.array(shap_values_xgb)
if shap_arr.ndim == 3:
    # Multiclass — average absolute SHAP values across all classes
    mean_shap = np.mean(np.abs(shap_arr), axis=2)
else:
    mean_shap = np.abs(shap_arr)

shap_df = pd.DataFrame(mean_shap, columns=clean_names)
shap_df.to_csv(ARTIFACTS_DIR / "shap_values.csv", index=False)
print("Saved shap_values.csv")

# ── Top Feature Summary ────────────────────────────────────────────────────────
mean_importance = shap_df.mean().sort_values(ascending=False)
print("\nTop 10 most important features (mean absolute SHAP value):")
print(mean_importance.head(10).to_string())

# ── SECTION 2 — SHAP on K-Means Clusters ──────────────────────────────────────
print("\n" + "="*60)
print("SECTION 2: SHAP Analysis on K-Means Clusters")
print("="*60)

# Clustering features used by Nathan
clustering_features = [
    "person_income",
    "loan_amnt",
    "loan_int_rate",
    "person_emp_length",
    "cb_person_cred_hist_length",
    "person_age"
]

# Clean names for cluster SHAP plots
cluster_clean_names = [f.replace("_", " ") for f in clustering_features]

# Load cleaned dataset for clustering
df_model = pd.read_csv(BASE_DIR / "data" / "processed" / "cleaned_dataset.csv")
X_cluster = df_model[clustering_features].copy()

# Remove outliers as Nathan did
for col in clustering_features:
    q1 = X_cluster[col].quantile(0.25)
    q3 = X_cluster[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    X_cluster = X_cluster[(X_cluster[col] >= lower) & (X_cluster[col] <= upper)]

X_cluster = X_cluster.reset_index(drop=True)
X_cluster_scaled = cluster_scaler.transform(X_cluster)

print(f"\nCluster sample size for SHAP: {len(X_cluster_scaled)} rows")
print(f"Number of clusters: {kmeans_model.n_clusters}")

# Use KernelExplainer for K-Means
# Use a small background sample and explain 300 rows
print("\nRunning KernelExplainer on K-Means clusters (this may take a minute)...")

background = shap.kmeans(X_cluster_scaled, 10)
cluster_predict_fn = lambda x: kmeans_model.predict(x).reshape(-1, 1).astype(float)

explainer_cluster = shap.KernelExplainer(cluster_predict_fn, background)

cluster_sample = X_cluster_scaled[:300]
shap_values_cluster = explainer_cluster.shap_values(cluster_sample)

print("SHAP values computed for K-Means clusters.")

# ── Cluster SHAP Bar Plot ──────────────────────────────────────────────────────
print("\nGenerating cluster SHAP bar plot...")

X_cluster_named = pd.DataFrame(cluster_sample, columns=cluster_clean_names)

plt.figure(figsize=(9, 5))
shap.summary_plot(
    shap_values_cluster,
    X_cluster_named,
    plot_type="bar",
    show=False
)
plt.title("Cluster Feature Importance — K-Means Credit Risk (Bar)", fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shap_cluster_bar.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved shap_cluster_bar.png")

# ── Save Cluster SHAP Values to CSV ───────────────────────────────────────────
print("\nSaving cluster SHAP values to CSV...")

shap_cluster_arr = np.array(shap_values_cluster)
if shap_cluster_arr.ndim == 3:
    shap_cluster_arr = shap_cluster_arr[:, :, 0]

shap_cluster_df = pd.DataFrame(
    np.abs(shap_cluster_arr),
    columns=cluster_clean_names
)
shap_cluster_df.to_csv(ARTIFACTS_DIR / "shap_cluster_values.csv", index=False)
print("Saved shap_cluster_values.csv")

# ── Final Summary ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SHAP Analysis Complete.")
print("="*60)

print("\nFiles saved to reports/figures/:")
print("  shap_bar.png              — global feature importance bar chart")
print("  shap_beeswarm.png         — global feature importance beeswarm")
print("  shap_waterfall.png        — local explanation for high-risk applicant")
print("  shap_cluster_bar.png      — cluster feature importance")

print("\nFiles saved to artifacts/:")
print("  shap_values.csv           — SHAP values for XGBoost classifier")
print("  shap_cluster_values.csv   — SHAP values for K-Means clusters")

print("\nTop 5 risk drivers for credit default:")
print(mean_importance.head(5).to_string())

print("\nCluster profiles:")
print(cluster_profiles.to_string())
