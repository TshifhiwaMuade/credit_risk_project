import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.stats import zscore
import joblib

# define paths
base_dir = Path(__file__).resolve().parent.parent
artifacts_dir = base_dir / "artifacts"
figures_dir = base_dir / "reports" / "figures"
data_path = base_dir / "data" / "processed" / "cleaned_dataset.csv"

# create folders
artifacts_dir.mkdir(exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

# load cleaned dataset
df = pd.read_csv(data_path)

# select clustering features
selected_features = [
    "person_income",
    "loan_amnt",
    "loan_int_rate",
    "person_emp_length",
    "cb_person_cred_hist_length",
    "person_age"
]

# remove outliers using IQR
for col in selected_features:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df = df[(df[col] >= lower) & (df[col] <= upper)]

# reset index after filtering
df = df.reset_index(drop=True)
print(f"rows after outlier removal: {len(df)}")

# subset data after outlier removal
X = df[selected_features].copy()

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# choose number of clusters (starting point)
k = 3

# train kmeans
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# evaluate clustering
score = silhouette_score(X_scaled, clusters)
print(f"silhouette score: {score:.4f}")

# save model and scaler
joblib.dump(kmeans, artifacts_dir / "kmeans_model.pkl")
joblib.dump(scaler, artifacts_dir / "cluster_scaler.pkl")

# add cluster column
df["cluster"] = clusters

# create cluster profiles
cluster_profiles = df.groupby("cluster")[selected_features].mean()

# derive cluster names dynamically from cluster center values
cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=selected_features
)
cluster_centers["cluster"] = range(kmeans.n_clusters)

# create risk score using z-scores for fair weighting
risk_cols = ["loan_int_rate", "loan_amnt", "person_income",
             "person_emp_length", "cb_person_cred_hist_length"]

cc = cluster_centers[risk_cols].copy()
cc_z = cc.apply(zscore)

cluster_centers["risk_score"] = (
     cc_z["loan_int_rate"]
   + cc_z["loan_amnt"]
   - cc_z["person_income"]
   - cc_z["person_emp_length"]
   - cc_z["cb_person_cred_hist_length"]
)

# sort clusters by risk
cluster_centers = cluster_centers.sort_values("risk_score")

# assign names
names = ["low risk", "medium risk", "high risk"]
cluster_name_map = {
    0: "high risk",
    1: "low risk",
    2: "medium risk"
}

# apply names
df["cluster_name"] = df["cluster"].map(cluster_name_map)

# add names to profiles
cluster_profiles["cluster_name"] = cluster_profiles.index.map(cluster_name_map)
cluster_profiles = cluster_profiles.reset_index()

# save outputs
df[["cluster", "cluster_name"]].to_csv("artifacts/cluster_labels.csv", index=False)
cluster_profiles.to_csv("artifacts/cluster_profiles.csv", index=False)

# display styled profile table in notebook
summary = cluster_profiles.copy()
summary.index = summary["cluster_name"]
summary = summary.drop(columns=["cluster", "cluster_name"])
summary.columns = [c.replace("_", " ").title() for c in summary.columns]

print("\ncluster profiles:")
print(cluster_profiles)

# pca visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centers_pca = pca.transform(kmeans.cluster_centers_)

palette = ["#2196F3", "#FF9800", "#F44336", "#4CAF50",
           "#9C27B0", "#00BCD4", "#FF5722", "#607D8B", "#E91E63"]
colors = {i: palette[i] for i in range(k)}
color_arr = [colors[c] for c in clusters]

plt.figure(figsize=(9, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=color_arr, alpha=0.4, s=10)

for cid, (x, y) in enumerate(centers_pca):
    label = cluster_name_map.get(cid, f"Cluster {cid}")
    plt.scatter(x, y, s=180, marker="X", color=colors[cid],
                edgecolors="black", linewidths=1, zorder=5)
    plt.annotate(label, (x, y), textcoords="offset points",
                 xytext=(8, 6), fontsize=9, fontweight="bold")

explained = pca.explained_variance_ratio_
plt.title("Credit Risk Clusters (PCA Projection)")
plt.xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
plt.tight_layout()
plt.savefig("reports/figures/cluster_pca.png", dpi=150)
plt.close()