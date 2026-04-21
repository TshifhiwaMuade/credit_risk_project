from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.stats import zscore
import joblib

# define paths
try:
    base_dir = Path(__file__).resolve().parent.parent
except NameError:
    # __file__ is not defined in Jupyter notebooks
    base_dir = Path.cwd().parent
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

# find optimal k using elbow method + silhouette scores
inertia = []
sil_scores = []
k_range = range(2, 10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(k_range, inertia, marker='o')
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia")

axes[1].plot(k_range, sil_scores, marker='o', color='orange')
axes[1].set_title("Silhouette Scores")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")

plt.tight_layout()
plt.savefig(figures_dir / "optimal_k.png")
plt.close()

optimal_k = k_range[sil_scores.index(max(sil_scores))]

# train kmeans
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init="auto")
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

# note: zscore is applied across k cluster centers only — with small k this
# is sensitive to extreme centers; revisit if k or data distribution changes
cc_z = cc.apply(zscore)

# person_age is included in clustering features to capture demographic segments,
# but excluded from the risk score — age alone is not a direct credit risk driver
# and including it could introduce age-based bias into the risk ranking.
cluster_centers["risk_score"] = (
     cc_z["loan_int_rate"]
   + cc_z["loan_amnt"]
   - cc_z["person_income"]
   - cc_z["person_emp_length"]
   - cc_z["cb_person_cred_hist_length"]
)

# sort clusters by risk
cluster_centers = cluster_centers.sort_values("risk_score")

# assign names based on sorted risk score
risk_labels = ["low risk", "medium risk", "high risk", "very high risk",
               "very low risk", "moderate-low risk", "moderate-high risk",
               "extreme risk", "minimal risk"]
names = risk_labels[:optimal_k]
cluster_name_map = {}

for i, cluster_id in enumerate(cluster_centers["cluster"]):
    cluster_name_map[cluster_id] = names[i]

# apply names
df["cluster_name"] = df["cluster"].map(cluster_name_map)

# add names to profiles
cluster_profiles["cluster_name"] = cluster_profiles.index.map(cluster_name_map)
cluster_profiles = cluster_profiles.reset_index()

# save outputs
df[["cluster", "cluster_name"]].to_csv(artifacts_dir / "cluster_labels.csv", index=False)
cluster_profiles.to_csv(artifacts_dir / "cluster_profiles.csv", index=False)

# display styled profile table in notebook
summary = cluster_profiles.copy()
summary.index = summary["cluster_name"]
summary = summary.drop(columns=["cluster", "cluster_name"])
summary.columns = [c.replace("_", " ").title() for c in summary.columns]

try:
    from IPython import get_ipython
    if get_ipython() is not None:
        display(summary.style
            .format("{:.2f}")
            .background_gradient(cmap="RdYlGn_r", axis=0)
            .set_caption("Cluster Profiles — Mean Feature Values"))
    else:
        raise RuntimeError("not in notebook")
except Exception:
    print("\nCluster Profiles — Mean Feature Values")
    print(summary.to_string(float_format="{:.2f}".format))

# pca visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centers_pca = pca.transform(kmeans.cluster_centers_)

palette = ["#2196F3", "#FF9800", "#F44336", "#4CAF50",
           "#9C27B0", "#00BCD4", "#FF5722", "#607D8B", "#E91E63"]
colors = {i: palette[i] for i in range(optimal_k)}
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
plt.savefig(figures_dir / "cluster_pca.png", dpi=150)
plt.close()