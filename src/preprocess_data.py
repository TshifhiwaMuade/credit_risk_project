from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import joblib


# 1. Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load cleaned dataset 
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Cleaned dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("\nLoaded dataset shape:", df.shape)


# 3. Separate features and target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]


# 4. Define categorical and numerical columns
categorical_cols = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file"
]

numerical_cols = [col for col in X.columns if col not in categorical_cols]


# 5. Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)


# 6. Apply preprocessing
X_processed = preprocessor.fit_transform(X)

print("\nPreprocessing complete")


# 7. Train / Validation / Test split (70/15/15 with stratify)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_processed,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print("\nData split complete")
print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)


# 8. SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nSMOTE applied to training set only")
print("Balanced train shape:", X_train.shape)


# 9. Save datasets
pd.DataFrame(X_train).to_csv(OUTPUT_DIR / "X_train.csv", index=False)
pd.DataFrame(X_val).to_csv(OUTPUT_DIR / "X_val.csv", index=False)
pd.DataFrame(X_test).to_csv(OUTPUT_DIR / "X_test.csv", index=False)

pd.DataFrame(y_train).to_csv(OUTPUT_DIR / "y_train.csv", index=False)
pd.DataFrame(y_val).to_csv(OUTPUT_DIR / "y_val.csv", index=False)
pd.DataFrame(y_test).to_csv(OUTPUT_DIR / "y_test.csv", index=False)

print("\nTrain/Val/Test datasets saved to data/processed/")


# 10. Save preprocessing pipeline
joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.pkl")

print("\nPreprocessor saved to artifacts/preprocessor.pkl")


print("\nFeature engineering pipeline complete. Data is ready for modelling.")
