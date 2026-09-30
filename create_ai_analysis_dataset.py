import pandas as pd
import random


# ============================================================
# AI ANALYSIS DATASET GENERATOR
# ============================================================

random.seed(42)

data = []


def add_sample(
    cpu_usage,
    cpu_temp,
    ram_usage,
    gpu_usage,
    gpu_temp,
    disk_usage,
    process_count,
    problem
):

    data.append([
        cpu_usage,
        cpu_temp,
        ram_usage,
        gpu_usage,
        gpu_temp,
        disk_usage,
        process_count,
        problem
    ])


# ============================================================
# 1. HEALTHY SYSTEM
# ============================================================

for _ in range(500):

    add_sample(
        round(random.uniform(5, 55), 1),
        round(random.uniform(35, 70), 1),
        round(random.uniform(20, 65), 1),
        round(random.uniform(0, 60), 1),
        round(random.uniform(35, 70), 1),
        round(random.uniform(20, 75), 1),
        random.randint(60, 180),
        "Healthy"
    )


# ============================================================
# 2. CPU PROBLEM
# ============================================================

for _ in range(300):

    add_sample(
        round(random.uniform(80, 99), 1),
        round(random.uniform(55, 78), 1),
        round(random.uniform(30, 70), 1),
        round(random.uniform(0, 60), 1),
        round(random.uniform(35, 70), 1),
        round(random.uniform(20, 75), 1),
        random.randint(150, 300),
        "CPU_Problem"
    )


# ============================================================
# 3. CPU THERMAL PROBLEM
# ============================================================

for _ in range(300):

    add_sample(
        round(random.uniform(55, 95), 1),
        round(random.uniform(80, 98), 1),
        round(random.uniform(30, 75), 1),
        round(random.uniform(0, 60), 1),
        round(random.uniform(35, 75), 1),
        round(random.uniform(20, 75), 1),
        random.randint(100, 280),
        "CPU_Thermal_Problem"
    )


# ============================================================
# 4. MEMORY PROBLEM
# ============================================================

for _ in range(300):

    add_sample(
        round(random.uniform(10, 65), 1),
        round(random.uniform(40, 70), 1),
        round(random.uniform(80, 98), 1),
        round(random.uniform(0, 60), 1),
        round(random.uniform(35, 70), 1),
        round(random.uniform(20, 75), 1),
        random.randint(150, 350),
        "Memory_Problem"
    )


# ============================================================
# 5. GPU PROBLEM
# ============================================================

for _ in range(300):

    add_sample(
        round(random.uniform(10, 65), 1),
        round(random.uniform(40, 70), 1),
        round(random.uniform(30, 70), 1),
        round(random.uniform(85, 99), 1),
        round(random.uniform(55, 78), 1),
        round(random.uniform(20, 75), 1),
        random.randint(80, 220),
        "GPU_Problem"
    )


# ============================================================
# 6. GPU THERMAL PROBLEM
# ============================================================

for _ in range(300):

    add_sample(
        round(random.uniform(20, 70), 1),
        round(random.uniform(40, 70), 1),
        round(random.uniform(30, 70), 1),
        round(random.uniform(60, 98), 1),
        round(random.uniform(80, 98), 1),
        round(random.uniform(20, 75), 1),
        random.randint(80, 250),
        "GPU_Thermal_Problem"
    )


# ============================================================
# 7. STORAGE PROBLEM
# ============================================================

for _ in range(300):

    add_sample(
        round(random.uniform(10, 65), 1),
        round(random.uniform(40, 70), 1),
        round(random.uniform(30, 70), 1),
        round(random.uniform(0, 60), 1),
        round(random.uniform(35, 70), 1),
        round(random.uniform(85, 99), 1),
        random.randint(60, 220),
        "Storage_Problem"
    )


# ============================================================
# 8. MULTIPLE PROBLEMS
# ============================================================

for _ in range(400):

    problem_type = random.choice([
        "CPU_RAM",
        "CPU_GPU",
        "RAM_GPU",
        "CPU_Thermal_RAM",
        "GPU_Thermal_RAM"
    ])

    if problem_type == "CPU_RAM":

        add_sample(
            random.uniform(85, 100),
            random.uniform(65, 82),
            random.uniform(85, 100),
            random.uniform(10, 60),
            random.uniform(40, 75),
            random.uniform(20, 75),
            random.randint(250, 450),
            "Multiple_Problems"
        )

    elif problem_type == "CPU_GPU":

        add_sample(
            random.uniform(85, 100),
            random.uniform(65, 82),
            random.uniform(30, 70),
            random.uniform(88, 100),
            random.uniform(65, 82),
            random.uniform(20, 75),
            random.randint(250, 450),
            "Multiple_Problems"
        )

    elif problem_type == "RAM_GPU":

        add_sample(
            random.uniform(10, 65),
            random.uniform(40, 70),
            random.uniform(85, 100),
            random.uniform(88, 100),
            random.uniform(65, 82),
            random.uniform(20, 75),
            random.randint(250, 450),
            "Multiple_Problems"
        )

    elif problem_type == "CPU_Thermal_RAM":

        add_sample(
            random.uniform(75, 100),
            random.uniform(82, 98),
            random.uniform(85, 100),
            random.uniform(10, 60),
            random.uniform(40, 75),
            random.uniform(20, 75),
            random.randint(250, 450),
            "Multiple_Problems"
        )

    else:

        add_sample(
            random.uniform(10, 65),
            random.uniform(40, 70),
            random.uniform(85, 100),
            random.uniform(70, 100),
            random.uniform(82, 98),
            random.uniform(20, 75),
            random.randint(250, 450),
            "Multiple_Problems"
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

columns = [
    "cpu_usage",
    "cpu_temp",
    "ram_usage",
    "gpu_usage",
    "gpu_temp",
    "disk_usage",
    "process_count",
    "problem"
]

df = pd.DataFrame(data, columns=columns)


# ============================================================
# SHUFFLE DATA
# ============================================================

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# SAVE DATASET
# ============================================================

output_file = "data/pc_ai_analysis_data.csv"

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# INFORMATION
# ============================================================

print("\n============================================")
print("AI ANALYSIS DATASET CREATED")
print("============================================")

print("\nDataset shape:")
print(df.shape)

print("\nProblem distribution:")
print(df["problem"].value_counts())

print("\nDataset saved to:")
print(output_file)

print("\nFirst 5 rows:")
print(df.head())