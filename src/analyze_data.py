import pandas as pd 

# Dataset load karna
df = pd.read_csv("data/pc_health_data.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nStaticstical sumary:")
print(df.describe())

print("\nRisk level count:")
print(df["risk_level"].value_counts())
