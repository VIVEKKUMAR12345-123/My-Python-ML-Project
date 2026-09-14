import psutil
import subprocess
import joblib
import pandas as pd 


# ==========================================
# 1. Load Trained AI Model
# ==========================================


model = joblib.load("models/pc_health_model.pkl")

print("Smart PC Health AI")
print("==================")
print("AI model loaded successfully!\n")


# ==========================================
# 2. Get GPU Information
# ==========================================


def get_gpu_info():

    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits"
            ],
            test=True
        )

        gpu_usage, gpu_temp = result.strip().split(",")

        return float(gpu_usage), float(gpu_temp)

    except Exception:
        return 0.0, 0.0


# ==========================================
# 3. Collect PC Health Data
# ==========================================


cpu_usage = psutil.cpu_percent(interval=1)

ram_usage = psutil.virtual_memory().percent

disk_usage = psutil.disk_usage("C:\\").percent

process_count = len(psutil.pids())

gpu_usage, gpu_temp = get_gpu_info()


# ==========================================
# 4. Display Current PC Data
# ==========================================


print("Current PC Health Data")
print("----------------------")

print(f"CPU Usage       : {cpu_usage:.1f}%")
print(f"RAM Usage       : {ram_usage:.1f}%")
print(f"GPU Usage       : {gpu_usage:.1f}%")
print(f"GPU Temperature : {gpu_temp:.1f}%")
print(f"Disk Usage      : {disk_usage:.1f}%")
print(f"Process Count   : {process_count}")


# ==========================================
# 5. CPU Temperature
# ==========================================


cpu_temp = 0.0

print(f"CPU Temperature : {cpu_temp:.1f}°C")


# ==========================================
# 6. Prepare Data for AI Model
# ==========================================


data = pd.DataFrame([[
    cpu_usage,
    cpu_temp,
    ram_usage,
    gpu_usage,
    gpu_temp,
    disk_usage,
    process_count
]], columns=[
    "cpu_usage",
    "cpu_temp",
    "ram_usage",
    "gpu_usage",
    "gpu_temp",
    "disk_usage",
    "process_count"
])


# ==========================================
# 7. AI Prediction
# ==========================================


prediction = model.predict(data)[0]


# ==========================================
# 8. Display AI Risk
# ==========================================


print("\nAI Risk Prediction")
print("------------------")
print(f"Risk Level: {prediction}")
