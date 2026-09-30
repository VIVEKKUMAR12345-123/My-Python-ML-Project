import pandas as pd
from pathlib import Path


# ============================================================
# LOAD DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "pc_ai_analysis_data.csv"

df = pd.read_csv(DATA_PATH)


# ============================================================
# STORAGE PROBLEM RANGE
# ============================================================

storage = df[df["problem"] == "Storage_Problem"]

multiple = df[df["problem"] == "Multiple_Problems"]


# ============================================================
# DISPLAY RANGES
# ============================================================

features = [
    "cpu_usage",
    "cpu_temp",
    "ram_usage",
    "gpu_usage",
    "gpu_temp",
    "disk_usage",
    "process_count"
]

print("\n============================================")
print("STORAGE vs MULTIPLE PROBLEMS")
print("============================================")

for feature in features:

    print(f"\n{feature}")

    print(
        "Storage Problem :",
        round(storage[feature].min(), 2),
        "to",
        round(storage[feature].max(), 2)
    )

    print(
        "Multiple Problems:",
        round(multiple[feature].min(), 2),
        "to",
        round(multiple[feature].max(), 2)
    )


# ============================================================
# MULTIPLE PROBLEM SAMPLES
# ============================================================

print("\n============================================")
print("MULTIPLE PROBLEM SAMPLE DATA")
print("============================================")

print(
    multiple[
        features + ["problem"]
    ].head(10).to_string(index=False)
)