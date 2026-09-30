import pandas as pd 
import joblib 

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "pc_ai_analysis_data.csv"
MODEL_PATH = BASE_DIR / "models" / "pc_ai_analysis_model.pkl"


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\n============================================")
print("AI ANALYSIS MODEL TRAINING")
print("============================================")

print("\nDataset shape:")
print(df.shape)

print("\nProblem distribution:")
print(df["problem"].value_counts())


# ============================================================
# FEATURES AND TARGET
# ===========================================================

features = [
    "cpu_usage",
    "cpu_temp",
    "ram_usage",
    "gpu_usage",
    "gpu_temp",
    "disk_usage",
    "process_count"
]

X = df[features]
y = df["problem"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_trian, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)


# ============================================================
# TRAIN MODEL
# ============================================================

model = DecisionTreeClassifier(
    random_state=42,
    max_depth=8
)

model.fit(X_train, y_trian)


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n============================================")
print("MODEL RESULTS")
print("============================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("\n============================================")
print("AI ANALYSIS MODEL SAVED")
print("============================================")

print("\nModel saved to:")
print(MODEL_PATH)