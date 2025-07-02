# ──────────────────────────────────────────────────────────────
# train_model.py
# Train a diabetes‑prediction model and save model + scaler
# ──────────────────────────────────────────────────────────────

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import pathlib

DATA_FILE          = "diabetes.csv"     # CSV filename
LIMIT_TO_70_ROWS   = False              # ← set True if you want only 70 rows
MODEL_FILE         = "diabetes_model.pkl"
SCALER_FILE        = "scaler.pkl"
TEST_SPLIT         = 0.20               # 20 % test data
RANDOM_STATE       = 42                 # reproducible splits & model

df = pd.read_csv(DATA_FILE)
if LIMIT_TO_70_ROWS and len(df) > 70:
    df = df.sample(n=70, random_state=RANDOM_STATE).reset_index(drop=True)
print(f"Dataset shape → {df.shape}")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = RandomForestClassifier(random_state=RANDOM_STATE)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {acc:.4f}")
print(classification_report(y_test, y_pred))


with open(MODEL_FILE, "wb")  as f:
    pickle.dump(model,  f)
with open(SCALER_FILE, "wb") as f:
    pickle.dump(scaler, f)

print(f"\nSaved model  → {pathlib.Path(MODEL_FILE).resolve()}")
print(f"Saved scaler → {pathlib.Path(SCALER_FILE).resolve()}")
