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
# TEST CASES
# ============================================================

test_cases = {

    "Healthy PC": [
        25,     # CPU Usage
        50,     # CPU Temperature
        50,     # RAM Usage
        20,     # GPU Usage
        45,     # GPU Temperature
        55,     # Disk Usage
        180     # Process Count
    ],

    "CPU Problem": [
        92,
        55,
        55,
        20,
        45,
        55,
        180
    ],

    "CPU Thermal Problem": [
        60,
        88,
        55,
        20,
        50,
        55,
        180
    ],

    "Memory Problem": [
        40,
        55,
        92,
        20,
        45,
        55,
        300
    ],

    "GPU Problem": [
        40,
        55,
        55,
        95,
        60,
        55,
        200
    ],

    "GPU Thermal Problem": [
        40,
        55,
        55,
        70,
        88,
        55,
        200
    ],

    "Storage Problem": [
        40,
        55,
        55,
        20,
        45,
        95,
        180
    ],

    "Multiple Problems": [
        95,
        90,
        92,
        96,
        90,
        65,
        350
    ]
}


# ============================================================
# TEST MODEL
# ============================================================

print("\n============================================")
print("AI ANALYSIS MODEL TEST")
print("============================================")

for test_name, values in test_cases.items():

    features = pd.DataFrame(
        [values],
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

    prediction = model.predict(features)[0]

    print(f"\n{test_name}")
    print("-" * 40)
    print(f"AI Prediction: {prediction}")


print("\n============================================")
print("TESTING COMPLETE")
print("============================================")