import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ==========================================
# 1. Load Dataset
# ==========================================


df = pd.read_csv("data/pc_health_data.csv")


# ==========================================
# 2. Features and Target
# ==========================================


X = df[
    [
        "cpu_usage",
        "cpu_temp",
        "ram_usage",
        "gpu_usage",
        "gpu_temp",
        "disk_usage",
        "process_count"
    ]
]

y = df["risk_level"]


# ==========================================
# 3. Train-Test-Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. Test Different Random Forest Settings
# ==========================================


settings = [
    (100, 5),
    (100, 7),
    (100, 10),
    (200, 7),
    (200, 10),
    (200, None)
]


print("Random Forest Model Comparison")
print("--------------------------------")

best_accuracy = 0
best_setting = None

for n_trees, depth in settings:

    model = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(
        f"Trees = {n_trees},"
        f"Max Depth = {depth} "
        f"--> Accuracy = {accuracy * 100:.2f}%"
    )

    if accuracy > best_accuracy:
        best_accuracy = accuracy 
        best_setting = (n_trees, depth)


print("\nBest Random Forest:")
print("Trees:", best_setting[0])
print("Max Depth:", best_setting[1])
print(f"Best Accuracy: {best_accuracy * 100:.2f}%")