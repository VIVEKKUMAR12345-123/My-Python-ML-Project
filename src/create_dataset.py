import pandas as pd
import random

data = []

for i in range(1500):

    cpu_usage = random.randint(10, 100)
    cpu_temp = random.randint(35, 95)

    ram_usage = random.randint(20, 100)

    gpu_usage = random.randint(0, 100)
    gpu_temp = random.randint(35, 90)

    disk_usage = random.randint(10, 100)
    process_count = random.randint(50, 250)

    # Risk score calculation karna
    risk_score = 0
     
    if cpu_usage >= 80:
        risk_score >= 2
    elif cpu_usage >= 60:
        risk_score +=1

    if cpu_temp >=85:
        risk_score += 3
    elif cpu_temp >= 70:
        risk_score += 2
    elif cpu_temp >= 60:
        risk_score +=1

    if ram_usage >= 90:
        risk_score += 2
    elif ram_usage >= 75:
        risk_score += 1

    if gpu_temp >= 82:
        risk_score += 3
    elif gpu_temp >= 70:
        risk_score += 2
    elif gpu_temp >= 60:
        risk_score += 1

    if gpu_usage >= 85:
        risk_score += 1

    if disk_usage >= 90:
        risk_score += 1

    if process_count >= 200:
        risk_score += 1

 
    # Final Risk Level
    if risk_score >= 6:
        risk_level = "Critical"

    elif risk_score >= 4:
        risk_level = "Warning"

    else:
        risk_level = "Normal"

    data.append([
        cpu_usage,
        cpu_temp,
        ram_usage,
        gpu_usage,
        gpu_temp,
        disk_usage,
        process_count,
        risk_level
    ])


columns = [
    "cpu_usage",
    "cpu_temp",
    "ram_usage",
    "gpu_usage",
    "gpu_temp",
    "disk_usage",
    "process_count",
    "risk_level"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("data/pc_health_data.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print("\nDataset shape:", df.shape)