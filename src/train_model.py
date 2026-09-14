import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


# ===========================================
# 1. Dataset load
# ===========================================


df = pd.read_csv("data/pc_health_data.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ===========================================
# 2. Separate Features (X) and Target (y)
# ===========================================


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

print("\nFeature (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

print("\nX shape:", X.shape)
print("y shape:", y.shape)


# ==============================================
# 3. Train-Test-Split 
# ==============================================


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# =============================================
# 4. Create Decision Tree Model
# =============================================


model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)


# =============================================
# 5. Train the Model
# =============================================


model.fit(X_train, y_train)

print("\nDecision Tree model trained successfully!")


# =============================================
# 6. Make predictions
# =============================================


y_pred = model.predict(X_test)

print("\nFirst 10 prediction:")
print(y_pred[:10])

print("\nActual values:")
print(y_test.iloc[:10].values)


# =============================================
# 7. Calculate Accuracy
# =============================================


accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("Model Accuracy (%):", accuracy * 100)


# =============================================
# 8. Classification Report
# =============================================


print('\nClassification Report:')
print(classification_report(y_test, y_pred))


# =============================================
# 9. Confusion Matrix
# =============================================

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))