import pandas as pd
import matplotlib.pyplot as plt

# Dataset load
df = pd.read_csv("data/pc_health_data.csv")

# Risk level distribution
df["risk_level"].value_counts().plot(kind="bar")

plt.title("PC Health Risk Level Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)

plt.show()