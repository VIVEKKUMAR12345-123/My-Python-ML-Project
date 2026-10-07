import pandas as pd 
import streamlit as st
import psutil
import subprocess
import joblib
import json
import re
import time
import platform
from datetime import datetime, timedelta
from pathlib import Path 


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart PC Health AI",
    page_icon="🖥️",
    layout="wide"
)


# ============================================================
# LOAD AI MODEL
# ============================================================


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "pc_health_model.pkl"

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

HISTORY_FILE = DATA_DIR / "pc_health_history.csv"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"AI Model could not be loaded: {e}")
    st.stop()


# ============================================================
# LOAD AI ANALYSIS MODEL
# ============================================================

AI_ANALYSIS_MODEL_PATH = BASE_DIR / "models" / "pc_ai_analysis_model.pkl"

try:
    ai_analysis_model = joblib.load(AI_ANALYSIS_MODEL_PATH)
except Exception as e:
    st.error(f"AI Analysis Model could not be loaded: {e}")
    st.stop()


# ============================================================
# FUNCTIONS
# ============================================================

def get_gpu_data():

    gpu_usage = None
    gpu_temperature = None

    try:

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=2
        )

        if result.returncode == 0:

            values = result.stdout.strip().split(",")

            if len(values) >= 2:
                gpu_usage = float(values[0].strip())
                gpu_temperature = float(values[1].strip())

    except Exception:
        pass

    return gpu_usage, gpu_temperature


def get_cpu_temperature():
    try:
        from HardwareMonitor.Util import OpenComputer

        computer = OpenComputer(cpu=True)
        computer.Update()

        temperatures = []

        for hardware in computer.Hardware:
            for sensor in hardware.Sensors:
                sensor_type = str(sensor.SensorType)

                if sensor_type == "Temperature":
                    if sensor.Value is not None:
                        name = str(sensor.Name)

                        # Prefer CPU package / Tctl-Tdie temperature
                        if (
                            "Tctl/Tdie" in name
                            or "Package" in name
                            or "CPU Package" in name
                        ):
                            temperatures.append(float(sensor.Value))

        if temperatures:
            return max(temperatures)

        return None

    except Exception:
        return None


# ============================================================
# PC COMPONENT INFORMATION
# ============================================================

def get_graphics_cards():
    graphics_cards = []

    try:
        result = subprocess.run(
            [
                "powershell",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name, AdapterRAM, DriverVersion, VideoProcessor | "
                "ConvertTo-Json"
            ],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0 or not result.stdout.strip():
            return graphics_cards

        data = json.loads(result.stdout)

        if isinstance(data, dict):
            data = [data]

        for gpu in data:
            name = gpu.get("Name", "Unknown")
            adapter_ram = gpu.get("AdapterRAM")
            driver = gpu.get("DriverVersion", "Unknown")
            processor = gpu.get("VideoProcessor", "Unknown")

            # Convert bytes to GB
            if adapter_ram:
                try:
                    vram_gb = round(int(adapter_ram) / (1024 ** 3), 2)
                except:
                    vram_gb = "Unknown"
            else:
                vram_gb = "Unknown"

            name_lower = name.lower()

            # Basic classification
            if any(x in name_lower for x in [
                "amd radeon(tm) graphics",
                "amd radeon graphics",
                "intel uhd",
                "intel iris",
                "intel hd graphics"
            ]):
                gpu_type = "Integrated"
            else:
                gpu_type = "Dedicated"

            graphics_cards.append({
                "Name": name,
                "Type": gpu_type,
                "VRAM": vram_gb,
                "Driver": driver,
                "Processor": processor
            })

    except Exception:
        return graphics_cards

    return graphics_cards


def get_gpu_usage_by_process():
    gpu_usage = {}

    try:
        result = subprocess.run(
            [
                "powershell",
                "-Command",
                "Get-CimInstance "
                "Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine | "
                "Select-Object Name, UtilizationPercentage | "
                "ConvertTo-Json -Compress"
            ],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0 or not result.stdout.strip():
            return gpu_usage

        data = json.loads(result.stdout)

        if isinstance(data, dict):
            data = [data]

        for item in data:

            name = item.get("Name", "")
            utilization = item.get("UtilizationPercentage", 0)

            match = re.search(r"pid_(\d+)", name, re.IGNORECASE)

            if match:
                pid = int(match.group(1))

                try:
                    utilization = float(utilization)
                except:
                    utilization = 0.0

                gpu_usage[pid] = gpu_usage.get(pid, 0.0) + utilization

    except Exception:
        return {}

    for pid in gpu_usage:
        gpu_usage[pid] = round(
            min(gpu_usage[pid], 100.0),
            1
        )

    return gpu_usage


def get_cpu_usage_by_process():

    cpu_usage = {}

    try:

        # Total logical CPU cores/threads
        cpu_count = psutil.cpu_count(logical=True) or 1

        processes = []

        # First measurement
        for proc in psutil.process_iter(["pid", "name"]):

            try:
                proc.cpu_percent(None)
                processes.append(proc)

            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
                psutil.ZombieProcess
            ):
                continue

        # CPU sampling interval
        time.sleep(0.5)

        # Second measurement
        for proc in processes:

            try:

                pid = proc.pid

                value = proc.cpu_percent(None)

                if value is None:
                    value = 0.0

                # Convert process CPU percentage
                # to total-system CPU percentage
                normalized_value = float(value) / cpu_count

                cpu_usage[pid] = round(
                    min(max(normalized_value, 0.0), 100.0),
                    1
                )

            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
                psutil.ZombieProcess
            ):
                continue

    except Exception:

        return {}

    return cpu_usage


def get_running_processes():

    processes = []

    cpu_usage_by_process = get_cpu_usage_by_process()

    try:
        gpu_usage_by_process = get_gpu_usage_by_process()
    except Exception:
        gpu_usage_by_process = {}

    # Windows system processes that are not useful
    # for the application monitoring table.
    system_processes = {
        "system",
        "system idle process",
        "registry",
        "smss.exe",
        "csrss.exe",
        "wininit.exe",
        "services.exe",
        "lsass.exe",
        "svchost.exe",
        "dwm.exe",
        "conhost.exe",
        "fontdrvhost.exe",
        "memory compression"
    }

    try:

        for proc in psutil.process_iter(
            ["pid", "name", "memory_info"]
        ):

            try:

                info = proc.info

                pid = info.get("pid")
                process_name = info.get("name") or "Unknown"

                if process_name.lower() in system_processes:
                    continue

                # RAM usage
                memory_info = info.get("memory_info")

                if memory_info:
                    ram_mb = round(
                        memory_info.rss / (1024 * 1024),
                        1
                    )
                else:
                    ram_mb = 0.0

                # CPU usage
                cpu_percent = cpu_usage_by_process.get(
                    pid,
                    0.0
                )

                # GPU usage
                gpu_percent = gpu_usage_by_process.get(
                    pid,
                    0.0
                )

                processes.append({
                    "PID": pid,
                    "Application": process_name,
                    "CPU %": round(
                        min(max(float(cpu_percent), 0.0), 100.0),
                        1
                    ),
                    "RAM (MB)": ram_mb,
                    "GPU %": round(
                        min(max(float(gpu_percent), 0.0), 100.0),
                        1
                    )
                })

            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
                psutil.ZombieProcess
            ):
                continue

    except Exception:
        return []

    # Show the most resource-intensive processes first
    processes.sort(
        key=lambda x: (
            x["CPU %"] +
            x["GPU %"] +
            (x["RAM (MB)"] / 1000)
        ),
        reverse=True
    )

    return processes


def get_actual_applications():
    processes = get_running_processes()

    applications = {}

    for proc in processes:

        name = proc["Application"]

        # Convert process name into a cleaner application name
        app_name = name

        if app_name.lower().endswith(".exe"):
            app_name = app_name[:-4]

        # Group same application processes together
        if app_name not in applications:

            applications[app_name] = {
                "Application": app_name,
                "Processes": 0,
                "CPU %": 0.0,
                "RAM (MB)": 0.0,
                "GPU %": 0.0
            }

        applications[app_name]["Processes"] += 1

        applications[app_name]["CPU %"] += proc["CPU %"]

        applications[app_name]["RAM (MB)"] += proc["RAM (MB)"]

        applications[app_name]["GPU %"] += proc["GPU %"]

    result = list(applications.values())

    for app in result:

        app["CPU %"] = round(
            min(app["CPU %"], 100.0),
            1
        )

        app["RAM (MB)"] = round(
            app["RAM (MB)"],
            1
        )

        app["GPU %"] = round(
            min(app["GPU %"], 100.0),
            1
        )

    result.sort(
        key=lambda x: x["RAM (MB)"],
        reverse=True
    )

    return result


def get_system_info():

    system_info = {}

    # ========================================================
    # CPU
    # ========================================================

    try:
        cpu_name = subprocess.run(
            [
                "powershell",
                "-Command",
                "(Get-CimInstance Win32_Processor).Name"
            ],
            capture_output=True,
            text=True,
            timeout=3
        ).stdout.strip()

        system_info["CPU"] = cpu_name if cpu_name else "Unknown"

    except Exception:
        system_info["CPU"] = "Unknown"

    system_info["CPU Cores"] = psutil.cpu_count(logical=False)
    system_info["CPU Threads"] = psutil.cpu_count(logical=True)

    try:
        cpu_freq = psutil.cpu_freq()

        if cpu_freq:
            system_info["CPU Frequency"] = (
                f"{cpu_freq.current / 1000:.2f} GHz"
            )
        else:
            system_info["CPU Frequency"] = "N/A"

    except Exception:
        system_info["CPU Frequency"] = "N/A"


    # ========================================================
    # RAM
    # ========================================================

    try:
        ram = psutil.virtual_memory()

        system_info["RAM"] = (
            f"{ram.total / (1024 ** 3):.2f} GB"
        )

        system_info["RAM Available"] = (
            f"{ram.available / (1024 ** 3):.2f} GB"
        )

    except Exception:
        system_info["RAM"] = "Unknown"
        system_info["RAM Available"] = "Unknown"


    # RAM TYPE

    try:
        ram_result = subprocess.run(
            [
                "powershell",
                "-Command",
                "(Get-CimInstance Win32_PhysicalMemory).SMBIOSMemoryType"
            ],
            capture_output=True,
            text=True,
            timeout=3
        )

        ram_type_value = (
            ram_result.stdout.strip().splitlines()[0]
            if ram_result.stdout.strip()
            else ""
        )

        ram_type_map = {
            "20": "DDR",
            "21": "DDR2",
            "22": "DDR2 FB-DIMM",
            "24": "DDR3",
            "26": "DDR4",
            "34": "DDR5"
        }

        system_info["RAM Type"] = ram_type_map.get(
            ram_type_value,
            "Unknown"
        )

    except Exception:
        system_info["RAM Type"] = "Unknown"

    # ========================================================
    # RAM SPEED
    # ========================================================

    try:
        ram_speed_result = subprocess.run(
            [
                "powershell",
                "-Command",
                "(Get-CimInstance Win32_PhysicalMemory).Speed"
            ],
            capture_output=True,
            text=True,
            timeout=3
        )

        speed_lines = [
            line.strip()
            for line in ram_speed_result.stdout.strip().splitlines()
            if line.strip()
        ]

        if speed_lines:
            speeds = []

            for speed in speed_lines:
                try:
                    speeds.append(int(float(speed)))
                except ValueError:
                    pass

            if speeds:
                unique_speeds = sorted(set(speeds))

                if len(unique_speeds) == 1:
                    system_info["RAM Speed"] = (
                        f"{unique_speeds[0]} MT/s"
                    )
                else:
                    system_info["RAM Speed"] = (
                        ", ".join(
                            f"{speed} MT/s"
                            for speed in unique_speeds
                        )
                    )
            else:
                system_info["RAM Speed"] = "N/A"

        else:
            system_info["RAM Speed"] = "N/A"

    except Exception:
        system_info["RAM Speed"] = "N/A"


    # ========================================================
    # GPU
    # ========================================================

    try:

        gpu_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=3
        )

        if gpu_result.returncode == 0 and gpu_result.stdout.strip():

            gpu_values = gpu_result.stdout.strip().split(",")

            if len(gpu_values) >= 2:

                system_info["GPU"] = gpu_values[0].strip()

                system_info["GPU VRAM"] = (
                    f"{gpu_values[1].strip()} MB"
                )

            else:
                system_info["GPU"] = "NVIDIA GPU"
                system_info["GPU VRAM"] = "Unknown"

        else:
            system_info["GPU"] = "Not detected"
            system_info["GPU VRAM"] = "N/A"


    except Exception:

        system_info["GPU"] = "Not detected"
        system_info["GPU VRAM"] = "N/A"


    # ========================================================
    # SSD / STORAGE
    # ========================================================

    try:

        disk_result = subprocess.run(
            [
                "powershell",
                "-Command",
                "Get-PhysicalDisk | "
                "Select-Object FriendlyName,MediaType,BusType,Size | "
                "ConvertTo-Csv -NoTypeInformation"
            ],
            capture_output=True,
            text=True,
            timeout=5
        )

        disk_lines = disk_result.stdout.strip().splitlines()

        if len(disk_lines) >= 2:

            disk_info = (
                disk_lines[1].strip('"').split('","')
            )

            if len(disk_info) >= 4:

                system_info["SSD Name"] = disk_info[0]
                system_info["SSD Type"] = disk_info[1]
                system_info["SSD Interface"] = disk_info[2]

            else:

                system_info["SSD Name"] = "Unknown"
                system_info["SSD Type"] = "Unknown"
                system_info["SSD Interface"] = "Unknown"

        else:

            system_info["SSD Name"] = "Unknown"
            system_info["SSD Type"] = "Unknown"
            system_info["SSD Interface"] = "Unknown"

    except Exception:

        system_info["SSD Name"] = "Unknown"
        system_info["SSD Type"] = "Unknown"
        system_info["SSD Interface"] = "Unknown"


    # C DRIVE

    try:

        disk = psutil.disk_usage("C:\\")

        system_info["Storage"] = (
            f"{disk.total / (1024 ** 3):.2f} GB"
        )

        system_info["Storage Free"] = (
            f"{disk.free / (1024 ** 3):.2f} GB"
        )

        system_info["Storage Used"] = (
            f"{disk.used / (1024 ** 3):.2f} GB"
        )

    except Exception:

        system_info["Storage"] = "Unknown"
        system_info["Storage Free"] = "Unknown"
        system_info["Storage Used"] = "Unknown"


    # ========================================================
    # MOTHERBOARD
    # ========================================================

    try:

        motherboard_result = subprocess.run(
            [
                "powershell",
                "-Command",
                "(Get-CimInstance Win32_BaseBoard).Manufacturer "
                "+ ' ' + "
                "(Get-CimInstance Win32_BaseBoard).Product"
            ],
            capture_output=True,
            text=True,
            timeout=3
        )

        motherboard = motherboard_result.stdout.strip()

        system_info["Motherboard"] = (
            motherboard if motherboard else "Unknown"
        )

    except Exception:

        system_info["Motherboard"] = "Unknown"


    # ========================================================
    # BIOS
    # ========================================================

    try:

        bios_result = subprocess.run(
            [
                "powershell",
                "-Command",
                "(Get-CimInstance Win32_BIOS).SMBIOSBIOSVersion"
            ],
            capture_output=True,
            text=True,
            timeout=3
        )

        bios = bios_result.stdout.strip()

        system_info["BIOS"] = (
            bios if bios else "Unknown"
        )

    except Exception:

        system_info["BIOS"] = "Unknown"


    # ========================================================
    # OPERATING SYSTEM
    # ========================================================

    system_info["Operating System"] = platform.system()
    system_info["OS Version"] = platform.release()
    system_info["Architecture"] = platform.architecture()[0]


    # IMPORTANT:
    # return must remain INSIDE get_system_info()

    return system_info


def get_uptime():

    boot_time = psutil.boot_time()

    current_time = time.time()

    uptime_seconds = current_time - boot_time

    uptime = str(timedelta(seconds=int(uptime_seconds)))

    return uptime


def get_pc_health():

    cpu_usage = psutil.cpu_percent(interval=0.5)

    cpu_temperature = get_cpu_temperature()

    ram_usage = psutil.virtual_memory().percent

    gpu_usage, gpu_temperature = get_gpu_data()

    disk_usage = psutil.disk_usage("C:\\").percent

    process_count = len(psutil.pids())

    uptime = get_uptime()
    

    return (
        cpu_usage,
        cpu_temperature,
        ram_usage,
        gpu_usage,
        gpu_temperature,
        disk_usage,
        process_count,
        uptime
    ) 


def predict_risk(
    cpu_usage,
    cpu_temperature,
    ram_usage,
    gpu_usage,
    gpu_temperature,
    disk_usage,
    process_count
):

    # Model was trained with 7 features
    features = [[
        cpu_usage,
        cpu_temperature,
        ram_usage,
        gpu_usage,
        gpu_temperature,
        disk_usage,
        process_count
    ]]

    prediction = model.predict(features)[0]

    return prediction


def predict_ai_analysis(
    cpu_usage,
    cpu_temperature,
    ram_usage,
    gpu_usage,
    gpu_temperature,
    disk_usage,
    process_count
):

    features = pd.DataFrame(
        [[
            cpu_usage,
            cpu_temperature,
            ram_usage,
            gpu_usage,
            gpu_temperature,
            disk_usage,
            process_count
        ]],
        columns=[
            "cpu_usage",
            "cpu_temp",
            "ram_usage",
            "gpu_usage",
            "gpu_temp",
            "disk_usage",
            "process_count"
        ]
    )

    prediction = ai_analysis_model.predict(features)[0]

    return prediction


# ============================================================
# SMART HEALTH ALERT SYSTEM
# ============================================================

def generate_health_alerts(
    cpu_usage,
    cpu_temperature,
    ram_usage,
    gpu_usage,
    gpu_temperature,
    disk_usage
):

    alerts = []

    # CPU Usage
    if cpu_usage >= 90:
        alerts.append(
            "🔴 CPU usage is extremely high. "
            "Close unnecessary applications and check background processes."
        )

    elif cpu_usage >= 75:
        alerts.append(
            "🟡 CPU usage is high. "
            "Monitor CPU-intensive applications."
        )


    # CPU Temperature
    if cpu_temperature is not None:

        if cpu_temperature >= 90:
            alerts.append(
                "🔴 CPU temperature is critically high. "
                "Check CPU cooling and airflow."
            )

        elif cpu_temperature >= 80:
            alerts.append(
                "🟡 CPU temperature is high. "
                "Monitor cooling performance."
            )


    # RAM
    if ram_usage >= 90:
        alerts.append(
            "🔴 RAM usage is critically high. "
            "Close unnecessary applications."
        )

    elif ram_usage >= 80:
        alerts.append(
            "🟡 RAM usage is high. "
            "Check applications consuming memory."
        )


    # GPU Usage
    if gpu_usage >= 95:
        alerts.append(
            "🟡 GPU usage is very high. "
            "This can be normal during gaming or GPU-intensive workloads."
        )


    # GPU Temperature
    if gpu_temperature >= 85:
        alerts.append(
            "🔴 GPU temperature is high. "
            "Check GPU cooling and airflow."
        )

    elif gpu_temperature >= 80:
        alerts.append(
            "🟡 GPU temperature is elevated. "
            "Monitor GPU temperature during heavy workloads."
        )


    # Disk
    if disk_usage >= 90:
        alerts.append(
            "🔴 Storage usage is critically high. "
            "Free up some disk space."
        )

    elif disk_usage >= 80:
        alerts.append(
            "🟡 Storage usage is high. "
            "Consider freeing some disk space."
        )

    return alerts


# ============================================================
# AI RISK EXPLANATION
# ============================================================

def generate_risk_explanation(
    cpu_usage,
    cpu_temperature,
    ram_usage,
    gpu_usage,
    gpu_temperature,
    disk_usage,
    risk_level
):

    reasons = []
    recommendations = []

    # CPU
    if cpu_usage >= 90:
        reasons.append(
            f"CPU usage is extremely high ({cpu_usage:.1f}%)."
        )
        recommendations.append(
            "Close unnecessary CPU-intensive applications."
        )

    elif cpu_usage >= 75:
        reasons.append(
            f"CPU usage is high ({cpu_usage:.1f}%)."
        )
        recommendations.append(
            "Monitor background and CPU-intensive processes."
        )


    # CPU Temperature
    if cpu_temperature is not None:

        if cpu_temperature >= 90:
            reasons.append(
                f"CPU temperature is critically high "
                f"({cpu_temperature:.1f} °C)."
            )
            recommendations.append(
                "Check CPU cooling, fan operation and airflow."
            )

        elif cpu_temperature >= 80:
            reasons.append(
                f"CPU temperature is elevated "
                f"({cpu_temperature:.1f} °C)."
            )
            recommendations.append(
                "Monitor CPU temperature during heavy workloads."
            )


    # RAM
    if ram_usage >= 90:
        reasons.append(
            f"RAM usage is critically high ({ram_usage:.1f}%)."
        )
        recommendations.append(
            "Close unnecessary applications to free memory."
        )

    elif ram_usage >= 80:
        reasons.append(
            f"RAM usage is high ({ram_usage:.1f}%)."
        )
        recommendations.append(
            "Check applications consuming large amounts of RAM."
        )


    # GPU
    if gpu_usage >= 95:
        reasons.append(
            f"GPU usage is very high ({gpu_usage:.1f}%)."
        )
        recommendations.append(
            "High GPU usage can be normal during gaming "
            "or GPU-intensive workloads."
        )


    # GPU Temperature
    if gpu_temperature >= 85:
        reasons.append(
            f"GPU temperature is high ({gpu_temperature:.1f} °C)."
        )
        recommendations.append(
            "Check GPU cooling and airflow."
        )

    elif gpu_temperature >= 80:
        reasons.append(
            f"GPU temperature is elevated ({gpu_temperature:.1f} °C)."
        )
        recommendations.append(
            "Monitor GPU temperature during heavy workloads."
        )


    # Disk
    if disk_usage >= 90:
        reasons.append(
            f"Disk usage is critically high ({disk_usage:.1f}%)."
        )
        recommendations.append(
            "Free up storage space."
        )

    elif disk_usage >= 80:
        reasons.append(
            f"Disk usage is high ({disk_usage:.1f}%)."
        )
        recommendations.append(
            "Consider freeing some disk space."
        )


    # AI prediction
    if risk_level == "Critical":

        reasons.append(
            "The Machine Learning model predicted a CRITICAL risk level."
        )

    elif risk_level == "Warning":

        reasons.append(
            "The Machine Learning model predicted a WARNING risk level."
        )

    # Normal system
    if not reasons:
        reasons.append(
            "No significant risk factors detected."
        )

    if not recommendations:
        recommendations.append(
            "No immediate action required. "
            "Your PC is operating normally."
        )

    return reasons, recommendations


# ============================================================
# HISTORICAL RISK ANALYSIS
# ============================================================


def get_risk_analysis():

    if not HISTORY_FILE.exists():
        return None

    try:
        history_data = pd.read_csv(HISTORY_FILE)

        # Remove accidental spaces from column names
        history_data.columns = history_data.columns.str.strip()

        if "AI Risk Level" not in history_data.columns:
            return None

        # Remove empty risk values
        risk_data = (
            history_data["AI Risk Level"]
            .astype(str)
            .str.strip()
        )

        risk_data = risk_data[
            risk_data.isin([
                "Normal",
                "Warning",
                "Critical"
            ])
        ]

        if len(risk_data) == 0:
            return None

        total = len(risk_data)

        normal_count = (risk_data == "Normal").sum()
        warning_count = (risk_data == "Warning").sum()
        critical_count = (risk_data == "Critical").sum()

        return {
            "total": total,
            "normal": normal_count,
            "warning": warning_count,
            "critical": critical_count
        }

    except Exception as e:
        return None



# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🖥️ Smart PC Health AI")

st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Overview",
        "📊 Performance",
        "🤖 AI Analysis",
        "🛡️ Security",
        "🖥️ System Info",
        "📈 History"
        
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "AI-Powered Real-Time PC Health & Risk Prediction System"
)


# ============================================================
# SECURITY PAGE
# ============================================================

if page == "🛡️ Security":

    st.title("🛡️ Security Monitoring")

    st.info(
        "Security monitoring module for basic system protection "
        "and security status information."
    )

    st.subheader("🔐 Security Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("🟢 System Monitoring Active")

    with col2:
        st.success("🟢 AI Risk Engine Active")

    with col3:
        st.success("🟢 Real-Time Monitoring Active")

    st.divider()

    st.subheader("🛡️ Security Overview")

    st.write(
        "This module provides a dedicated area for monitoring "
        "system health and security-related conditions."
    )

    st.warning(
        "⚠️ Advanced security checks can be added in the next phase."
    )



# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Overview":

    st.title("🖥️ Smart PC Health AI")

    st.markdown(
        "### AI-Powered Real-Time PC Health & Risk Prediction System"
    )

    st.info(
        "Monitor your PC in real time, analyze hardware performance, "
        "and predict system health risk using Machine Learning."
    )

    st.divider()


# ============================================================
# INITIALIZE SESSION HISTORY
# ============================================================

if "history" not in st.session_state:

    if HISTORY_FILE.exists():

        st.session_state.history = pd.read_csv(
            HISTORY_FILE
        )

    else:

        st.session_state.history = pd.DataFrame(
            columns=[
                "Time",
                "CPU Usage",
                "CPU Temperature",
                "RAM Usage",
                "GPU Usage",
                "GPU Temperature",
                "Disk Usage",
                "Process Count",
                "AI Risk Level"
            ]
        )


def group_processes_by_application(processes):

    applications = {}

    for proc in processes:

        name = proc["Application"]

        # Remove .exe
        if name.lower().endswith(".exe"):
            app_name = name[:-4]
        else:
            app_name = name

        if app_name not in applications:

            applications[app_name] = {
                "Application": app_name,
                "Processes": 0,
                "CPU %": 0.0,
                "RAM (MB)": 0.0,
                "GPU %": 0.0
            }

        applications[app_name]["Processes"] += 1

        applications[app_name]["CPU %"] += proc["CPU %"]

        applications[app_name]["RAM (MB)"] += proc["RAM (MB)"]

        applications[app_name]["GPU %"] += proc["GPU %"]

    result = list(applications.values())

    for app in result:

        app["CPU %"] = round(
            min(app["CPU %"], 100.0),
            1
        )

        app["RAM (MB)"] = round(
            app["RAM (MB)"],
            1
        )

        app["GPU %"] = round(
            min(app["GPU %"], 100.0),
            1
        )

    result.sort(
        key=lambda x: (
            x["CPU %"] +
            x["GPU %"]
        ),
        reverse=True
    )

    return result
    

@st.fragment(run_every="1s")
def realtime_dashboard():

    (
        cpu_usage,
        cpu_temperature,
        ram_usage,
        gpu_usage,
        gpu_temperature,
        disk_usage,
        process_count,
        uptime
    ) = get_pc_health()

    ai_analysis = predict_ai_analysis(
        cpu_usage,
        cpu_temperature,
        ram_usage,
        gpu_usage,
        gpu_temperature,
        disk_usage,
        process_count
    )

    cpu_temp_display = (
        f"{cpu_temperature:.1f} °C"
        if cpu_temperature is not None
        else "N/A"
    )


    # ========================================================
    # AI PREDICTION
    # ========================================================

    risk_level = predict_risk(
        cpu_usage,
        cpu_temperature,
        ram_usage,
        gpu_usage,
        gpu_temperature,
        disk_usage,
        process_count
    )


    if page == "🏠 Overview":
        st.title("🏠 PC Health Overview")

        st.markdown(
        "Monitor your PC health, performance and AI risk status in real time."
        )

        st.divider()

    elif page == "📊 Performance":
        st.title("📊 Performance Monitoring")

        # RUNNING APPLICATIONS

        st.markdown("### 🖥️ Running Applications")

        if "process_data" not in st.session_state:
            st.session_state.process_data = get_running_processes()

        if st.button("🔄 Refresh Applications"):
            st.session_state.process_data = get_running_processes()

        running_processes = st.session_state.process_data

        running_applications = group_processes_by_application(
            running_processes
        )

        # GROUP SAME APPLICATIONS
        application_data = {}

        for process in running_processes:

            app_name = process["Application"]

            if app_name not in application_data:
                application_data[app_name] = {
                    "Application": app_name,
                    "CPU %": 0,
                    "RAM (MB)": 0,
                    "GPU %": 0
                }

            application_data[app_name]["CPU %"] += process["CPU %"]
            application_data[app_name]["RAM (MB)"] += process["RAM (MB)"]
            application_data[app_name]["GPU %"] += process["GPU %"]

        running_applications = list(application_data.values())
        
        if running_processes:

            process_df = pd.DataFrame(running_applications)

            total_apps = len(process_df)

            top_cpu = process_df.loc[
                process_df["CPU %"].idxmax()
            ]

            top_ram = process_df.loc[
                process_df["RAM (MB)"].idxmax()
            ]

            top_gpu = process_df.loc[
                process_df["GPU %"].idxmax()
            ]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "🖥️ Running Applications",
                    total_apps
                )

            with col2:
                st.metric(
                    "⚙️ Highest CPU",
                    f"{top_cpu['CPU %']}%"
                )
                st.caption(top_cpu["Application"])

            with col3:
                st.metric(
                    "🧠 Highest RAM",
                    f"{top_ram['RAM (MB)']:.1f} MB"
                )
                st.caption(top_ram["Application"])

            with col4:
                st.metric(
                    "🎮 Highest GPU",
                    f"{top_gpu['GPU %']}%"
                )
                st.caption(top_gpu["Application"])


        if running_applications:

            process_df = pd.DataFrame(running_applications)

            process_df["RAM (MB)"] = process_df["RAM (MB)"].round(1)

            st.dataframe(
                process_df,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info("No running application information available.")

    elif page == "🤖 AI Analysis":
        st.title("🤖 AI Health Analysis")


    # ============================================================
    # PROFESSIONAL KPI CARDS
    # ============================================================

    if page == "🏠 Overview":

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            st.metric(
                "🖥️ CPU",
                f"{cpu_usage:.1f}%"
            )

        with kpi2:
            st.metric(
                "🧠 RAM",
                f"{ram_usage:.1f}%"
            )

        with kpi3:
            st.metric(
                "🎮 GPU",
                f"{gpu_usage:.1f}%"
            )

        with kpi4:
            st.metric(
                "🤖 AI Risk",
                risk_level
            )

        st.divider()


        # ============================================================
        # SYSTEM STATUS
        # ============================================================

        st.subheader("📊 Current System Status")

        status_col1, status_col2, status_col3 = st.columns(3)

        with status_col1:
            if cpu_usage < 75:
                st.success("🟢 CPU Performance: Normal")
            elif cpu_usage < 90:
                st.warning("🟡 CPU Performance: High")
            else:
                st.error("🔴 CPU Performance: Critical")

        with status_col2:
            if ram_usage < 80:
                st.success("🟢 Memory Usage: Normal")
            elif ram_usage < 90:
                st.warning("🟡 Memory Usage: High")
            else:
                st.error("🔴 Memory Usage: Critical")

        with status_col3:
            if risk_level == "Normal":
                st.success("🟢 AI Health Status: Normal")
            elif risk_level == "Warning":
                st.warning("🟡 AI Health Status: Warning")
            else:
                st.error("🔴 AI Health Status: Critical")

        st.divider()


    # ========================================================
    # SMART ALERTS
    # ========================================================

    alerts = generate_health_alerts(
        cpu_usage,
        cpu_temperature,
        ram_usage,
        gpu_usage,
        gpu_temperature,
        disk_usage
    )

    reasons, recommendations = generate_risk_explanation(
        cpu_usage,
        cpu_temperature,
        ram_usage,
        gpu_usage,
        gpu_temperature,
        disk_usage,
        risk_level
    )


    # ========================================================
    # SAVE COMPLETE MONITORING DATA
    # ========================================================

    new_data = pd.DataFrame(
        [[
            datetime.now(),
            cpu_usage,
            cpu_temperature,
            ram_usage,
            gpu_usage,
            gpu_temperature,
            disk_usage,
            process_count,
            risk_level
        ]],
        columns=[
            "Time",
            "CPU Usage",
            "CPU Temperature",
            "RAM Usage",
            "GPU Usage",
            "GPU Temperature",
            "Disk Usage",
            "Process Count",
            "AI Risk Level"
        ]
    )

    # Save data permanently to CSV
    if HISTORY_FILE.exists():

        new_data.to_csv(
            HISTORY_FILE,
            mode="a",
            header=False,
            index=False
        )

    else:

        new_data.to_csv(
            HISTORY_FILE,
            mode="w",
            header=True,
            index=False
        )

    # ========================================================
    # SAFETY CHECK FOR SESSION HISTORY
    # ========================================================

    if "history" not in st.session_state:

        st.session_state.history = pd.DataFrame(
            columns=[
                "Time",
                "CPU Usage",
                "CPU Temperature",
                "RAM Usage",
                "GPU Usage",
                "GPU Temperature",
                "Disk Usage",
                "Process Count",
                "AI Risk Level"
            ]
        )

    # Update live graph history
    st.session_state.history = pd.concat(
        [st.session_state.history, new_data],
        ignore_index=True
    )

    # Keep only latest 60 readings
    st.session_state.history = st.session_state.history.tail(60)


    # ========================================================
    # AI RISK & HEALTH ANALYSIS
    # ========================================================

    if page == "🤖 AI Analysis":

        # ========================================================
        # PROFESSIONAL AI RISK CARD
        # ========================================================

        st.subheader("🤖 AI Risk Assessment")

        if risk_level == "Normal":
            st.success(
                "🟢 **AI Risk Level: NORMAL**\n\n"
                "Your system is currently operating within a healthy range."
            )

        elif risk_level == "Warning":
            st.warning(
                "🟡 **AI Risk Level: WARNING**\n\n"
                "Some system parameters are elevated. Monitoring is recommended."
            )

        else:
            st.error(
                "🔴 **AI Risk Level: CRITICAL**\n\n"
                "The AI model has detected potentially critical system conditions."
            )
        

        # ========================================================
        # AI PROBLEM DIAGNOSIS
        # ========================================================

        st.markdown("### 🧠 AI Problem Diagnosis")

        ai_analysis_names = {
            "Healthy": "🟢 Healthy",
            "CPU_Problem": "🔴 CPU Performance Problem",
            "CPU_Thermal_Problem": "🌡️ CPU Thermal Problem",
            "Memory_Problem": "🟠 Memory / RAM Problem",
            "GPU_Problem": "🔴 GPU Performance Problem",
            "GPU_Thermal_Problem": "🌡️ GPU Thermal Problem",
            "Storage_Problem": "💾 Storage / Disk Problem",
            "Multiple_Problems": "⚠️ Multiple System Problems"
        }

        ai_analysis_display = ai_analysis_names.get(
            ai_analysis,
            ai_analysis
        )

        st.info(
            f"**AI Diagnosis:** {ai_analysis_display}"
        )


        # ========================================================
        # AI DIAGNOSIS DETAILS
        # ========================================================

        diagnosis_factors = []

        if ram_usage >= 80:
            diagnosis_factors.append(
                f"RAM usage is high ({ram_usage:.1f}%)."
            )

        if cpu_usage >= 75:
            diagnosis_factors.append(
                f"CPU usage is high ({cpu_usage:.1f}%)."
            )

        if cpu_temperature >= 80:
            diagnosis_factors.append(
                f"CPU temperature is high ({cpu_temperature:.1f} °C)."
            )

        if gpu_usage >= 80:
            diagnosis_factors.append(
                f"GPU usage is high ({gpu_usage:.1f}%)."
            )

        if gpu_temperature >= 80:
            diagnosis_factors.append(
                f"GPU temperature is high ({gpu_temperature:.1f} °C)."
            )

        if disk_usage >= 85:
            diagnosis_factors.append(
                f"Disk usage is high ({disk_usage:.1f}%)."
            )

        if process_count >= 250:
            diagnosis_factors.append(
                f"Process count is high ({process_count})."
            )

        if diagnosis_factors:

            st.markdown("#### 🔍 Detected Factors")

            for factor in diagnosis_factors:
                st.write(f"• {factor}")

        else:

            st.success(
                "🟢 No significant system factors detected."
            )


        # ========================================================
        # AI DIAGNOSIS RECOMMENDATIONS
        # ========================================================

        diagnosis_recommendations = []

        if ai_analysis == "Healthy":

            diagnosis_recommendations.append(
                "No immediate action required. Your PC is operating normally."
            )

        elif ai_analysis == "CPU_Problem":

            diagnosis_recommendations.append(
                "Check applications using high CPU resources."
            )
            diagnosis_recommendations.append(
                "Close unnecessary background applications to reduce CPU load."
            )

        elif ai_analysis == "CPU_Thermal_Problem":

            diagnosis_recommendations.append(
                "Check CPU cooling and airflow."
            )
            diagnosis_recommendations.append(
                "Avoid sustained heavy CPU workloads until temperature decreases."
            )

        elif ai_analysis == "Memory_Problem":

            diagnosis_recommendations.append(
                "Close unnecessary applications using large amounts of RAM."
            )
            diagnosis_recommendations.append(
                "Check which running applications are consuming the most memory."
            )

        elif ai_analysis == "GPU_Problem":

            diagnosis_recommendations.append(
                "Check applications using high GPU resources."
            )
            diagnosis_recommendations.append(
                "Close unnecessary GPU-intensive applications if they are not required."
            )

        elif ai_analysis == "GPU_Thermal_Problem":

            diagnosis_recommendations.append(
                "Check GPU cooling and airflow."
            )
            diagnosis_recommendations.append(
                "Reduce sustained GPU workload until the temperature decreases."
            )

        elif ai_analysis == "Storage_Problem":

            diagnosis_recommendations.append(
                "Check available disk space and remove unnecessary files."
            )
            diagnosis_recommendations.append(
                "Monitor disk usage if storage utilization remains high."
            )

        elif ai_analysis == "Multiple_Problems":

            if ram_usage >= 80:

                diagnosis_recommendations.append(
                    f"RAM usage is high ({ram_usage:.1f}%). "
                    "Close unnecessary memory-intensive applications."
                )

            if cpu_usage >= 75:

                diagnosis_recommendations.append(
                    f"CPU usage is high ({cpu_usage:.1f}%). "
                    "Check applications using high CPU resources."
                )

            if cpu_temperature >= 80:

                diagnosis_recommendations.append(
                    f"CPU temperature is high ({cpu_temperature:.1f} °C). "
                    "Check CPU cooling and airflow."
                )

            if gpu_usage >= 80:

                diagnosis_recommendations.append(
                    f"GPU usage is high ({gpu_usage:.1f}%). "
                    "Check which applications are using the GPU."
                )

            if gpu_temperature >= 80:

                diagnosis_recommendations.append(
                    f"GPU temperature is high ({gpu_temperature:.1f} °C). "
                    "Check GPU cooling and airflow."
                )

            if disk_usage >= 85:

                diagnosis_recommendations.append(
                    f"Disk usage is high ({disk_usage:.1f}%). "
                    "Check available storage space."
                )

            if process_count >= 250:

                diagnosis_recommendations.append(
                    f"Process count is high ({process_count}). "
                    "Close unnecessary background applications."
                )

            if not diagnosis_recommendations:

                diagnosis_recommendations.append(
                    "Multiple system conditions require monitoring."
                )

        if diagnosis_recommendations:

            st.markdown("#### 💡 AI Recommendations")

            for recommendation in diagnosis_recommendations:
                st.write(f"• {recommendation}")


        with st.expander("🚨 Smart Health Alerts", expanded=False):

            if alerts:

                for alert in alerts:
                    st.warning(alert)

            else:

                st.success(
                    "🟢 No health issues detected. "
                    "Your PC is operating normally."
                )


        with st.expander("🔍 AI Risk Explanation", expanded=False):

            if reasons:

                st.markdown("### ⚠️ Detected Factors")

                for reason in reasons:
                    st.write(f"• {reason}")

            else:

                st.success(
                    "🟢 No significant risk factors detected."
                )

        st.divider()


    if page == "🏠 Overview":

        # ========================================================
        # CURRENT PC HEALTH
        # ========================================================

    
        st.subheader("📊 Current PC Health Monitoring")


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "🖥️ CPU Usage",
                f"{cpu_usage:.1f}%"
            )

            st.metric(
                "🌡️ CPU Temperature",
                cpu_temp_display
            )


        with col2:

            st.metric(
                "💾 RAM Usage",
                f"{ram_usage:.1f}%"
            )

            st.metric(
                "💽 Disk Usage",
                f"{disk_usage:.1f}%"
            )


        with col3:

            st.metric(
                "🎮 GPU Usage",
                f"{gpu_usage:.1f}%"
            )

            st.metric(
                "🌡️ GPU Temperature",
                f"{gpu_temperature:.1f} °C"
            )


        with col4:

            st.metric(
                "⚙️ Process Count",
                process_count
            )

            st.metric(
                "⏱️ System Uptime",
                uptime
            )

        ## LIVE PEPFORMANCE HISTORY

        st.subheader("📈 Live Performance Monitoring History")

        chart_data = st.session_state.history.set_index("Time")

        st.line_chart(
            chart_data[
                [
                    "CPU Usage",
                    "CPU Temperature",
                    "RAM Usage",
                    "GPU Usage",
                    "GPU Temperature",
                    "Disk Usage"
                ]
            ]
        )

        st.divider()


        # ========================================================
        # SYSTEM DATA
        # ========================================================

        st.subheader("📋 System Data")


        system_data = {

            "CPU Usage": f"{cpu_usage:.1f}%",

            "CPU Temperature": cpu_temp_display,

            "RAM Usage": f"{ram_usage:.1f}%",

            "GPU Usage": f"{gpu_usage:.1f}%",

            "GPU Temperature": f"{gpu_temperature:.1f} °C",

            "Disk Usage": f"{disk_usage:.1f}%",

            "Process Count": process_count,

            "System Uptime": uptime,

            "AI Risk Level": risk_level,

            "Last Updated": datetime.now().strftime(
                "%H:%M:%S"
            )
        }


        st.table(system_data)


# ============================================================
# RUN DASHBOARD
# ============================================================

if page in ["🏠 Overview", "📊 Performance", "🤖 AI Analysis"]:
    realtime_dashboard()


# ============================================================
# PC COMPONENTS
# ============================================================

if page == "🖥️ System Info":

    st.divider()

    st.subheader("🧩 PC Components & System Specifications")

    system_info = get_system_info()

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🧠 Processor")

        st.write(
            f"**CPU:** {system_info['CPU']}"
        )

        st.write(
            f"**Cores:** {system_info['CPU Cores']}"
        )

        st.write(
            f"**Threads:** {system_info['CPU Threads']}"
        )

        st.write(
            f"**Frequency:** {system_info['CPU Frequency']}"
        )
        

        st.markdown("### 🎮 Graphics Cards")

        graphics_cards = get_graphics_cards()

        if graphics_cards:

            for i, gpu in enumerate(graphics_cards, start=1):

                st.markdown(f"#### 🎮 Graphics Card {i}")

                st.write(f"**Name:** {gpu['Name']}")
                st.write(f"**Type:** {gpu['Type']}")

                st.write(
                    f"**VRAM:** {gpu['VRAM']} GB"
                    if isinstance(gpu["VRAM"], (int, float))
                    else "**VRAM:** Unknown"
                )

                st.write(f"**Driver Version:** {gpu['Driver']}")
                st.write(f"**Video Processor:** {gpu['Processor']}")

        else:
            st.info("No graphics card information available.")


    with col2:

        st.markdown("### 💾 Memory")

        st.write(
            f"**Total RAM:** {system_info['RAM']}"
        )

        st.write(
            f"**RAM Type:** {system_info['RAM Type']}"
        )

        st.write(
            f"**RAM Speed:** {system_info['RAM Speed']}"
        )

        st.write(
            f"**Available RAM:** {system_info['RAM Available']}"
        )


        st.markdown("### 💽 Storage")

        st.write(
            f"**SSD:** {system_info['SSD Name']}"
        )

        st.write(
            f"**Type:** {system_info['SSD Type']}"
        )

        st.write(
            f"**Interface:** {system_info['SSD Interface']}"
        )

        st.write(
            f"**Total Storage:** {system_info['Storage']}"
        )

        st.write(
            f"**Free Space:** {system_info['Storage Free']}"
        )


        st.markdown("### 🖥️ Motherboard")

        st.write(
            f"**Motherboard:** {system_info['Motherboard']}"
        )


        st.markdown("### 🔧 BIOS")

        st.write(
            f"**BIOS Version:** {system_info['BIOS']}"
        )


        st.markdown("### 🪟 Operating System")

        st.write(
            f"**OS:** {system_info['Operating System']} "
            f"{system_info['OS Version']}"
        )

        st.write(
            f"**Architecture:** {system_info['Architecture']}"
        )


# ============================================================
# RISK HISTORY / TIMELINE
# ============================================================

def get_risk_timeline():

    try:

        history_data = pd.read_csv(HISTORY_FILE)

        history_data.columns = (
            history_data.columns.str.strip()
        )

        if (
            "Time" not in history_data.columns
            or "AI Risk Level" not in history_data.columns
        ):
            return None

        history_data["Time"] = pd.to_datetime(
            history_data["Time"],
            errors="coerce"
        )

        history_data["AI Risk Level"] = (
            history_data["AI Risk Level"]
            .astype(str)
            .str.strip()
        )

        history_data = history_data.dropna(
            subset=["Time"]
        )

        history_data = history_data[
            history_data["AI Risk Level"].isin(
                [
                    "Normal",
                    "Warning",
                    "Critical"
                ]
            )
        ]

        if history_data.empty:
            return None

        # Risk score for graph
        risk_mapping = {
            "Normal": 0,
            "Warning": 1,
            "Critical": 2
        }

        history_data["Risk Score"] = (
            history_data["AI Risk Level"]
            .map(risk_mapping)
        )

        # Latest 120 readings
        timeline = history_data.tail(300).copy()

        # Latest risk status
        current_risk = (
            timeline.iloc[-1]["AI Risk Level"]
        )

        return {
            "timeline": timeline,
            "current_risk": current_risk
        }

    except Exception:

        return None


# ============================================================
# LIVE HISTORICAL RISK ANALYSIS
# ============================================================

@st.fragment(run_every="2s")
def historical_risk_dashboard():

    st.divider()

    st.subheader("📊 Historical Risk Analysis")

    risk_analysis = get_risk_analysis()

    if risk_analysis is not None:

        total = risk_analysis["total"]
        normal = risk_analysis["normal"]
        warning = risk_analysis["warning"]
        critical = risk_analysis["critical"]

        normal_percent = (normal / total) * 100
        warning_percent = (warning / total) * 100
        critical_percent = (critical / total) * 100


        # ====================================================
        # RISK METRICS
        # ====================================================

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "📋 Total Readings",
                total
            )

        with col2:

            st.metric(
                "🟢 Normal",
                f"{normal_percent:.1f}%"
            )

        with col3:

            st.metric(
                "🟡 Warning",
                f"{warning_percent:.1f}%"
            )

        with col4:

            st.metric(
                "🔴 Critical",
                f"{critical_percent:.1f}%"
            )


        # ====================================================
        # RISK DISTRIBUTION
        # ====================================================

        risk_chart = pd.DataFrame(
            {
                "Risk Level": [
                    "Normal",
                    "Warning",
                    "Critical"
                ],

                "Readings": [
                    normal,
                    warning,
                    critical
                ]
            }
        )

        st.bar_chart(
            risk_chart.set_index("Risk Level")
        )


        # ====================================================
        # RISK TIMELINE
        # ====================================================

        st.markdown("### 📈 Risk Timeline")

        risk_timeline = get_risk_timeline()

        if risk_timeline is not None:

            timeline_data = risk_timeline["timeline"]
            current_risk = risk_timeline["current_risk"]


            # ====================================================
            # CURRENT RISK STATUS
            # ====================================================

            if current_risk == "Normal":

                st.success(
                    "🟢 Current Risk Status: NORMAL"
                )

            elif current_risk == "Warning":

                st.warning(
                    "🟡 Current Risk Status: WARNING"
                )

            elif current_risk == "Critical":

                st.error(
                    "🔴 Current Risk Status: CRITICAL"
                )


            # ====================================================
            # TIMELINE GRAPH
            # ====================================================

            timeline_chart = (
                timeline_data[
                    ["Time", "Risk Score"]
                ]
                .set_index("Time")
            )

            st.line_chart(
                timeline_chart,
                height=350
            )

            st.caption(
                "Risk Level: "
                "0 = 🟢 Normal | "
                "1 = 🟡 Warning | "
                "2 = 🔴 Critical"
            )

        else:

            st.info(
                "Not enough historical risk data available yet."
            )



# ============================================================
# RUN DASHBOARDS
# ============================================================

if page == "📈 History":
    historical_risk_dashboard()


# ============================================================
# DOWNLOAD HISTORY
# ============================================================

if page == "📈 History":

    st.divider()

    st.subheader("📥 Download Monitoring History")

    if HISTORY_FILE.exists():

        with open(HISTORY_FILE, "rb") as file:

            st.download_button(
                label="📥 Download PC Health History (CSV)",
                data=file,
                file_name="pc_health_history.csv",
                mime="text/csv"
            )

    else:

        st.info("No monitoring history available yet.") 


    # ============================================================
    # CLEAR HISTORY
    # ============================================================

    st.subheader("🗑️ Reset Monitoring History")

    # Initialize confirmation state
    if "confirm_clear_history" not in st.session_state:
        st.session_state.confirm_clear_history = False

    # Show clear button
    if HISTORY_FILE.exists():

        if not st.session_state.confirm_clear_history:

            if st.button("🗑️ Clear All History"):

                st.session_state.confirm_clear_history = True
                st.rerun()

        else:

            st.warning(
                "⚠️ Are you sure you want to clear all monitoring history?"
            )

            col1, col2 = st.columns(2)

            with col1:
                if st.button("✅ Yes, Clear History"):

                    # Delete history file
                    HISTORY_FILE.unlink()

                    # Reset session history
                    st.session_state.history = pd.DataFrame(
                        columns=[
                            "Time",
                            "CPU Usage",
                            "CPU Temperature",
                            "RAM Usage",
                            "GPU Usage",
                            "GPU Temperature",
                            "Disk Usage",
                            "Process Count",
                            "AI Risk Level"
                        ]
                    )

                    # Reset confirmation state
                    st.session_state.confirm_clear_history = False

                    # Store success message
                    st.session_state.history_cleared = True

                    st.rerun()

            with col2:
                if st.button("❌ Cancel"):

                    st.session_state.confirm_clear_history = False
                    st.rerun()

    # Show success message after rerun
    if st.session_state.get("history_cleared", False):

        st.success("✅ Monitoring history has been cleared successfully.")

        # Remove message after displaying it
        st.session_state.history_cleared = False

    else:

        if not HISTORY_FILE.exists():
            st.info("No monitoring history available to clear.")

            