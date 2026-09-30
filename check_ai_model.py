import joblib
import pandas as pd
from pathlib import Path


# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "pc_ai_analysis_model.pkl"

model = joblib.load(MODEL_PATH)


# ============================================================
# MULTIPLE PROBLEM TEST
# ============================================================

features = pd.DataFrame(
    [[
        95,   # CPU Usage
        90,   # CPU Temperature
        92,   # RAM Usage
        96,   # GPU Usage
        90,   # GPU Temperature
        95,   # Disk Usage
        350   # Process Count
    ]],
    columns=[
        "cpu_usage",
        "cpu_temp",
        "ram_usage",
        "gpu_usage",
        "gpu_temp",
        "disk_usage",
        "process_count"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(features)[0]

print("\n============================================")
print("AI MODEL ANALYSIS")
print("============================================")

print("\nInput Values:")
print(features.to_string(index=False))

print("\nPrediction:")
print(prediction)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n============================================")
print("FEATURE IMPORTANCE")
print("============================================")

importance = pd.DataFrame({
    "Feature": features.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print(importance.to_string(index=False))


# ============================================================
# TREE INFORMATION
# ============================================================

print("\n============================================")
print("MODEL INFORMATION")
print("============================================")

print("Model Type:", type(model).__name__)
print("Tree Depth:", model.get_depth())
print("Number of Leaves:", model.get_n_leaves())