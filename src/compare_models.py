import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
# 4. Test Different Tree Depths
# ==========================================


depths = [3, 5, 6, 7, 8, 10]

print("Decision Tree Model Comparison")
print("--------------------------------")

for depth in depths:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(
        f"max_depth = {depth} --> "
        f"Accuracy = {accuracy * 100:2f}%"
    )
