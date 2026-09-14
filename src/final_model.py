import pandas as pd 
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import(
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. Load Dataset
# ==========================================


df = pd.read_csv("data/pc_health_data.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


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
# 3. Train-Test Split
# ==========================================


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. Best Random Forest Model
# ==========================================


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# ==========================================
# 5. Train Model
# ==========================================


model.fit(X_train, y_train)

print("\nFinal Random Forest trained successfully!")


# ==========================================
# 6. Prediction
# ==========================================

y_pred= model.predict(X_test)


# ==========================================
# 7. Accuracy
# ==========================================


accuracy = accuracy_score(y_test, y_pred)

print("\nFinal Model Accuracy:")
print(f"{accuracy * 100:.2f}%")


# ==========================================
# 8. Classification Report
# ==========================================


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# 9. Confusion Matrix
# ==========================================


print("Confusion Matrix:")

cm = confusion_matrix(y_test, y_pred)

print(cm)


# ==========================================
# 10. Save Model
# ==========================================


joblib.dump(model, "models/pc_health_model.pkl")

print("\nModel saved successfully!")
print("Location: models/pc_health_model.pkl")

