import pandas as pd 

from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report 


# ==========================================
# 1. Load Dataset
# ==========================================


df = pd.read_csv("data/pc_health_data.csv")

print("Dataset loaded sucessfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. Features (X) and Target (y)
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
# 4. Create Random Forest Model
# ==========================================


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=7
)


# ==========================================
# 5. Train Model
# ==========================================


model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")


# ==========================================
# 6. Prediction
# ==========================================


y_pred = model.predict(X_test)


# ==========================================
# 7. Accuracy
# ==========================================


accuracy = accuracy_score(y_test, y_pred)

print("\nRandom Forest Accuracy:")
print(f"{accuracy * 100:.2f}%")


# ==========================================
# 8. Classification Report
# ==========================================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))