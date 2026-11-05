"""CodeNova Task 4: customer segmentation pipeline."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "shopping_trends.csv").drop_duplicates()
numeric = ["Age", "Purchase Amount (USD)", "Review Rating", "Previous Purchases"]
categorical = ["Gender", "Category", "Season", "Subscription Status",
               "Discount Applied", "Promo Code Used", "Payment Method",
               "Shipping Type", "Frequency of Purchases"]
for col in numeric:
    df[col] = pd.to_numeric(df[col], errors="coerce")
for col in numeric:
    df[col] = df[col].fillna(df[col].median())
X = df[numeric + categorical]
prep = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])
Z = prep.fit_transform(X)
Z = Z.toarray() if hasattr(Z, "toarray") else Z
ks, inertias, sils = range(2, 9), [], []
rng = np.random.RandomState(42)
idx = rng.choice(len(Z), min(1500, len(Z)), replace=False)
for k in ks:
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    lab = model.fit_predict(Z)
    inertias.append(model.inertia_)
    sils.append(silhouette_score(Z[idx], lab[idx]))
best_k = list(ks)[int(np.argmax(sils))]
model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
df["Cluster"] = model.fit_predict(Z)
coords = PCA(n_components=2, random_state=42).fit_transform(Z)
df["PCA1"], df["PCA2"] = coords[:, 0], coords[:, 1]
(ROOT / "reports").mkdir(exist_ok=True)
df.to_csv(ROOT / "reports" / "customer_segments.csv", index=False)
print(f"Selected k={best_k}; silhouette={max(sils):.4f}")\nprint("Interpret cluster profiles carefully: the available data may separate mainly on discount/promo indicators.")
