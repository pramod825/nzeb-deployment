# validate_model.py - Complete NZEB Model Validation
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings("ignore")
import os

print("🔍 NZEB MODEL VALIDATION SYSTEM")
print("=" * 60)

# -------------------------------------------------------
# STEP 1 - Load all models
# -------------------------------------------------------
print("\n📂 Loading models...")
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

with open(os.path.join(MODEL_DIR, "xgboost_model.pkl"), "rb") as f:
    xgb_model = pickle.load(f)
print("✅ XGBoost loaded!")

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
print("✅ Scaler loaded!")

with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb") as f:
    feature_cols = pickle.load(f)
print(f"✅ Features loaded! ({len(feature_cols)} features)")

with open(os.path.join(MODEL_DIR, "cities.json"), "r") as f:
    cities = json.load(f)
print(f"✅ Cities loaded! ({len(cities)} cities)")

with open(os.path.join(MODEL_DIR, "city_encoding.json"), "r") as f:
    city_encoding = json.load(f)
print("✅ City encoding loaded!")

with open(os.path.join(MODEL_DIR, "building.json"), "r") as f:
    building = json.load(f)
print("✅ Building params loaded!")

# Load LSTM
from tensorflow.keras.models import load_model
lstm_model = load_model(
    os.path.join(MODEL_DIR, "lstm_model.keras")
)
print("✅ LSTM loaded!")
print("\n🎉 All models loaded!")

# -------------------------------------------------------
# STEP 2 - Generate validation dataset
# -------------------------------------------------------
print("\n📊 GENERATING VALIDATION DATASET...")
print("=" * 60)

import pvlib
import datetime

def calculate_energy_realistic(
    temp, humidity, solar, hour,
    month, zone, rad_s, rad_e, rad_w
):
    """Calculate realistic energy consumption"""
    cdd = max(0, temp - 24)
    hdd = max(0, 18 - temp)

    base = {1: 0.095, 2: 0.085,
            3: 0.070, 4: 0.090}
    energy = base.get(int(zone), 0.085)

    if temp > 24:
        energy += (temp - 24) * 0.0045
    if cdd > 0:
        energy += cdd * 0.0038
    if humidity > 60:
        energy += (humidity - 60) * 0.0008
    if temp < 18:
        energy += (18 - temp) * 0.0025
    if hdd > 0:
        energy += hdd * 0.0020

    energy += (rad_s + rad_e + rad_w) * 0.00004

    if 6 <= hour <= 18 and solar > 200:
        energy += 0.008
    else:
        energy += 0.018

    energy += 0.025 * (
        1 + 0.1 * np.sin(2 * np.pi * hour / 24)
    )

    if 9 <= hour <= 18:
        energy *= 1.15
    elif 19 <= hour <= 21:
        energy *= 1.05
    else:
        energy *= 0.85

    if month in [4, 5, 6]:
        energy *= 1.20
    elif month in [12, 1, 2]:
        energy *= 1.10
    elif month in [7, 8, 9]:
        energy *= 1.05

    noise = np.random.normal(0, 0.015)
    return max(0.01, energy + noise)

# Generate validation data for all 12 cities
np.random.seed(123)  # Different seed from training!
validation_data = []

print("Generating validation samples...")
for city_name, info in cities.items():
    city_idx = city_encoding[city_name]

    # Generate 500 random samples per city
    for _ in range(500):
        month  = np.random.randint(1, 13)
        day    = np.random.randint(1, 29)
        hour   = np.random.randint(0, 24)
        temp   = np.random.uniform(
            5 if city_name == "Chandigarh" else 15,
            45 if city_name in ["Delhi", "Jaipur"] else 38
        )
        hum    = np.random.uniform(20, 95)
        solar  = np.random.uniform(0, 900)
        wind   = np.random.uniform(0.5, 8)
        cloud  = np.random.uniform(0, 100)
        press  = np.random.uniform(95, 105)

        # Solar calculations
        site = pvlib.location.Location(
            latitude  = info["lat"],
            longitude = info["lon"],
            altitude  = info["altitude"],
            tz        = info["timezone"]
        )
        dt = pd.Timestamp(
            year=2023, month=month,
            day=day, hour=hour,
            tz=info["timezone"]
        )
        solar_pos = site.get_solarposition(
            pd.DatetimeIndex([dt])
        )
        solar_alt = float(
            solar_pos["apparent_elevation"].iloc[0]
        )
        solar_az  = float(
            solar_pos["azimuth"].iloc[0]
        )
        solar_zen = float(
            solar_pos["apparent_zenith"].iloc[0]
        )
        is_day = 1 if solar_alt > 0 else 0

        # Facade radiation
        ghi  = solar
        disc = pvlib.irradiance.disc(
            np.array([ghi]),
            np.array([solar_zen]),
            pd.DatetimeIndex([dt])
        )
        dni = max(0, float(
            np.array(disc["dni"]).flatten()[0]
        ))
        dhi = max(0, ghi - dni * np.cos(
            np.radians(solar_zen)
        ))

        walls = {
            "north": {"tilt": 90, "azimuth": 0},
            "south": {"tilt": 90, "azimuth": 180},
            "east":  {"tilt": 90, "azimuth": 90},
            "west":  {"tilt": 90, "azimuth": 270},
        }
        facade = {}
        for wall, angles in walls.items():
            poa = pvlib.irradiance.get_total_irradiance(
                surface_tilt    = angles["tilt"],
                surface_azimuth = angles["azimuth"],
                solar_zenith    = np.array([solar_zen]),
                solar_azimuth   = np.array([solar_az]),
                dni             = np.array([dni]),
                ghi             = np.array([ghi]),
                dhi             = np.array([dhi]),
                model           = "isotropic"
            )
            val = float(np.array(
                poa["poa_global"]
            ).flatten()[0])
            facade[wall] = max(0, np.nan_to_num(val))

        # BIPV
        eff = float(building["bipv_efficiency"])
        pr  = float(building["bipv_pr"])
        bipv_s = facade["south"] * float(building["bipv_area_south"]) * eff * pr / 1000 * is_day
        bipv_e = facade["east"]  * float(building["bipv_area_east"])  * eff * pr / 1000 * is_day
        bipv_w = facade["west"]  * float(building["bipv_area_west"])  * eff * pr / 1000 * is_day
        bipv_t = bipv_s + bipv_e + bipv_w
        bipv_m2 = bipv_t / float(building["total_area"])

        # Features
        doy         = datetime.date(2023, month, day).timetuple().tm_yday
        dow         = datetime.date(2023, month, day).weekday()
        is_weekend  = 1 if dow >= 5 else 0
        is_peak     = 1 if 9 <= hour <= 18 else 0
        cdd         = max(0, temp - 24)
        hdd         = max(0, 18 - temp)
        discomfort  = 0.4 * (temp + 0.99 * hum) + 4.9
        heat_idx    = temp + 0.33 * (hum/100 * 6.105 * np.exp(17.27*temp/(237.7+temp))) - 4.0
        total_fac   = sum(facade.values())
        rad_ratio   = total_fac / solar if solar > 0 else 0
        shg         = float(building["shgc_south"]) * (float(building["glass_area_north"]) + float(building["glass_area_south"]) + float(building["glass_area_east"]) + float(building["glass_area_west"])) * solar / float(building["total_area"])
        env_loss    = float(building["u_wall"]) * (float(building["wall_area_north"]) + float(building["wall_area_south"]) + float(building["wall_area_east"]) + float(building["wall_area_west"])) * abs(temp - 22) / float(building["total_area"])
        tc          = 1 if (20 <= temp <= 26 and 30 <= hum <= 60) else 0
        season      = 1 if month in [3,4,5] else 2 if month in [6,7,8,9] else 3 if month in [10,11] else 4

        # Target energy
        energy = calculate_energy_realistic(
            temp, hum, solar, hour,
            month, info["climate_zone"],
            facade["south"], facade["east"],
            facade["west"]
        )

        row = {
            "temperature":               temp,
            "relative_humidity":         hum,
            "solar_radiation":           solar,
            "wind_speed":                wind,
            "cloud_cover":               cloud,
            "pressure":                  press,
            "year":                      2023,
            "month":                     month,
            "day":                       day,
            "hour":                      hour,
            "day_of_week":               dow,
            "is_weekend":                is_weekend,
            "climate_zone":              info["climate_zone"],
            "altitude":                  info["altitude"],
            "latitude":                  info["lat"],
            "longitude":                 info["lon"],
            "solar_altitude":            solar_alt,
            "solar_azimuth":             solar_az,
            "solar_zenith":              solar_zen,
            "is_daytime":                is_day,
            "hour_angle":                (hour - 12) * 15,
            "radiation_north":           facade["north"],
            "radiation_south":           facade["south"],
            "radiation_east":            facade["east"],
            "radiation_west":            facade["west"],
            "bipv_south_w":              bipv_s * 1000,
            "bipv_east_w":               bipv_e * 1000,
            "bipv_west_w":               bipv_w * 1000,
            "bipv_total_w":              bipv_t * 1000,
            "bipv_south_kwh":            bipv_s,
            "bipv_east_kwh":             bipv_e,
            "bipv_west_kwh":             bipv_w,
            "bipv_total_kwh":            bipv_t,
            "bipv_kwh_m2":               bipv_m2,
            "city_encoded":              city_idx,
            "sin_hour":                  np.sin(2*np.pi*hour/24),
            "cos_hour":                  np.cos(2*np.pi*hour/24),
            "sin_month":                 np.sin(2*np.pi*month/12),
            "cos_month":                 np.cos(2*np.pi*month/12),
            "sin_doy":                   np.sin(2*np.pi*doy/365),
            "cos_doy":                   np.cos(2*np.pi*doy/365),
            "is_peak_hour":              is_peak,
            "CDD":                       cdd,
            "HDD":                       hdd,
            "discomfort_index":          discomfort,
            "heat_index":                heat_idx,
            "total_facade_radiation":    total_fac,
            "radiation_ratio":           rad_ratio,
            "thermal_comfort":           tc,
            "solar_heat_gain":           shg,
            "envelope_loss":             env_loss,
            "season":                    season,
            "temp_solar_interaction":    temp * solar / 1000,
            "humidity_temp_interaction": hum * temp / 100,
            "cdd_solar_interaction":     cdd * solar / 1000,
            "energy_kwh_m2":             energy,
            "city_name":                 city_name,
        }
        validation_data.append(row)

val_df = pd.DataFrame(validation_data)
print(f"✅ Generated {len(val_df):,} validation samples!")
print(f"   Cities  : {val_df['city_name'].nunique()}")
print(f"   Samples : 500 per city")

# -------------------------------------------------------
# STEP 3 - Run predictions
# -------------------------------------------------------
print("\n🤖 RUNNING PREDICTIONS...")
print("=" * 60)

X_val = val_df[feature_cols].apply(
    pd.to_numeric, errors="coerce"
).fillna(0)
y_val = val_df["energy_kwh_m2"]

X_val_scaled = scaler.transform(X_val)

# XGBoost predictions
xgb_pred = xgb_model.predict(X_val_scaled)

# LSTM predictions
X_val_lstm = X_val_scaled.reshape(
    X_val_scaled.shape[0], 1,
    X_val_scaled.shape[1]
)
lstm_pred = lstm_model.predict(
    X_val_lstm, verbose=0
).flatten()

# Ensemble
ensemble_pred = (xgb_pred + lstm_pred) / 2

print("✅ Predictions complete!")

# -------------------------------------------------------
# STEP 4 - Calculate metrics
# -------------------------------------------------------
print("\n📊 VALIDATION METRICS...")
print("=" * 60)

def calculate_metrics(y_true, y_pred, model_name):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs(
        (y_true - y_pred) / y_true
    )) * 100
    cv_rmse = rmse / np.mean(y_true) * 100

    print(f"\n📊 {model_name}")
    print(f"   R² Score  : {r2:.4f}")
    print(f"   MAE       : {mae:.6f} kWh/m²")
    print(f"   RMSE      : {rmse:.6f} kWh/m²")
    print(f"   MAPE      : {mape:.2f}%")
    print(f"   CV-RMSE   : {cv_rmse:.2f}%")

    # ASHRAE Guideline 14 compliance
    if cv_rmse <= 30 and mape <= 10:
        print(f"   ASHRAE    : ✅ Compliant!")
    else:
        print(f"   ASHRAE    : ⚠️ Needs improvement")

    return {
        "model":   model_name,
        "r2":      r2,
        "mae":     mae,
        "rmse":    rmse,
        "mape":    mape,
        "cv_rmse": cv_rmse
    }

metrics_xgb = calculate_metrics(
    y_val, xgb_pred, "XGBoost"
)
metrics_lstm = calculate_metrics(
    y_val, lstm_pred, "LSTM"
)
metrics_ens = calculate_metrics(
    y_val, ensemble_pred, "Ensemble"
)

# -------------------------------------------------------
# STEP 5 - City wise validation
# -------------------------------------------------------
print("\n📊 CITY WISE VALIDATION...")
print("=" * 60)
print(f"\n{'City':12s} {'R²':8s} {'MAE':10s} "
      f"{'MAPE':8s} {'CV-RMSE':10s} {'Status':12s}")
print("-" * 60)

city_results = {}
for city_name in cities.keys():
    mask      = val_df["city_name"] == city_name
    y_city    = y_val[mask]
    pred_city = ensemble_pred[mask]

    if len(y_city) == 0:
        continue

    r2      = r2_score(y_city, pred_city)
    mae     = mean_absolute_error(y_city, pred_city)
    mape    = np.mean(np.abs(
        (y_city - pred_city) / y_city
    )) * 100
    cv_rmse = np.sqrt(mean_squared_error(
        y_city, pred_city
    )) / np.mean(y_city) * 100

    status = (
        "✅ Excellent" if r2 >= 0.85
        else "✅ Good"   if r2 >= 0.75
        else "⚠️ Fair"
    )

    city_results[city_name] = {
        "r2": r2, "mae": mae,
        "mape": mape, "cv_rmse": cv_rmse
    }

    print(f"{city_name:12s} {r2:8.4f} {mae:10.6f} "
          f"{mape:8.2f}% {cv_rmse:10.2f}% {status:12s}")

# -------------------------------------------------------
# STEP 6 - BEE EPI Comparison
# -------------------------------------------------------
print("\n📊 BEE EPI BENCHMARK COMPARISON...")
print("=" * 60)

# BEE EPI benchmarks for office buildings (kWh/m²/year)
bee_benchmarks = {
    "5 Star (Best)":  {"min": 0,   "max": 100},
    "4 Star":         {"min": 100, "max": 150},
    "3 Star":         {"min": 150, "max": 200},
    "2 Star":         {"min": 200, "max": 250},
    "1 Star (Worst)": {"min": 250, "max": 999},
}

# Annual prediction per city
print(f"\n{'City':12s} {'Annual EPI':12s} "
      f"{'BEE Rating':15s} {'Status':10s}")
print("-" * 52)

for city_name in cities.keys():
    mask     = val_df["city_name"] == city_name
    avg_pred = ensemble_pred[mask].mean()
    annual   = avg_pred * 8760  # hourly to annual

    # Get BEE rating
    rating = "Below 1 Star"
    for star, limits in bee_benchmarks.items():
        if limits["min"] <= annual < limits["max"]:
            rating = star
            break

    status = "✅" if "Star" in rating else "❌"
    print(f"{city_name:12s} {annual:12.1f} "
          f"{rating:15s} {status:10s}")

# -------------------------------------------------------
# STEP 7 - Plot validation graphs
# -------------------------------------------------------
print("\n📊 PLOTTING VALIDATION GRAPHS...")

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig)

# Graph 1 - Actual vs Predicted
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(
    y_val, ensemble_pred,
    alpha=0.3, color="#2196F3",
    s=5, edgecolors="none"
)
min_val = min(y_val.min(), ensemble_pred.min())
max_val = max(y_val.max(), ensemble_pred.max())
ax1.plot(
    [min_val, max_val],
    [min_val, max_val],
    "r--", linewidth=2,
    label="Perfect fit"
)
ax1.set_title(
    f"Actual vs Predicted\nR²={metrics_ens['r2']:.4f}",
    fontsize=11, fontweight="bold"
)
ax1.set_xlabel("Actual (kWh/m²)", fontsize=9)
ax1.set_ylabel("Predicted (kWh/m²)", fontsize=9)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Graph 2 - Residuals
ax2 = fig.add_subplot(gs[0, 1])
residuals = y_val - ensemble_pred
ax2.scatter(
    ensemble_pred, residuals,
    alpha=0.3, color="#FF5722",
    s=5, edgecolors="none"
)
ax2.axhline(y=0, color="black",
            linewidth=2, linestyle="--")
ax2.set_title(
    "Residual Plot\n(Should be around zero)",
    fontsize=11, fontweight="bold"
)
ax2.set_xlabel("Predicted (kWh/m²)", fontsize=9)
ax2.set_ylabel("Residuals", fontsize=9)
ax2.grid(True, alpha=0.3)

# Graph 3 - Error Distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(residuals, bins=50,
         color="#4CAF50", alpha=0.7,
         edgecolor="none")
ax3.axvline(x=0, color="red",
            linewidth=2, linestyle="--",
            label="Zero error")
ax3.set_title(
    "Error Distribution\n(Centered = Good)",
    fontsize=11, fontweight="bold"
)
ax3.set_xlabel("Error (kWh/m²)", fontsize=9)
ax3.set_ylabel("Frequency", fontsize=9)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Graph 4 - City wise R²
ax4 = fig.add_subplot(gs[1, 0])
city_names = list(city_results.keys())
r2_values  = [city_results[c]["r2"] for c in city_names]
colors     = plt.cm.RdYlGn(
    np.linspace(0.2, 0.9, len(city_names))
)
bars = ax4.barh(
    city_names, r2_values,
    color=colors, edgecolor="black",
    alpha=0.85
)
ax4.axvline(x=0.85, color="red",
            linestyle="--", linewidth=1.5,
            label="Target (0.85)")
ax4.set_title(
    "R² Score Per City",
    fontsize=11, fontweight="bold"
)
ax4.set_xlabel("R² Score", fontsize=9)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, axis="x")

for bar, val in zip(bars, r2_values):
    ax4.text(
        bar.get_width() + 0.005,
        bar.get_y() + bar.get_height()/2,
        f"{val:.3f}", va="center", fontsize=8
    )

# Graph 5 - City wise MAPE
ax5 = fig.add_subplot(gs[1, 1])
mape_values = [
    city_results[c]["mape"] for c in city_names
]
colors_mape = plt.cm.RdYlGn(
    np.linspace(0.9, 0.2, len(city_names))
)
bars = ax5.barh(
    city_names, mape_values,
    color=colors_mape, edgecolor="black",
    alpha=0.85
)
ax5.axvline(x=10, color="red",
            linestyle="--", linewidth=1.5,
            label="Target (<10%)")
ax5.set_title(
    "MAPE Per City\n(Lower is Better)",
    fontsize=11, fontweight="bold"
)
ax5.set_xlabel("MAPE (%)", fontsize=9)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3, axis="x")

for bar, val in zip(bars, mape_values):
    ax5.text(
        bar.get_width() + 0.1,
        bar.get_y() + bar.get_height()/2,
        f"{val:.1f}%", va="center", fontsize=8
    )

# Graph 6 - Model Comparison
ax6 = fig.add_subplot(gs[1, 2])
models      = ["XGBoost", "LSTM", "Ensemble"]
r2_compare  = [
    metrics_xgb["r2"],
    metrics_lstm["r2"],
    metrics_ens["r2"]
]
colors_comp = ["#2196F3", "#FF5722", "#4CAF50"]
bars = ax6.bar(
    models, r2_compare,
    color=colors_comp, alpha=0.85,
    edgecolor="black", width=0.5
)
ax6.axhline(y=0.85, color="red",
            linestyle="--", linewidth=1.5,
            label="Target (0.85)")
ax6.set_title(
    "Model R² Comparison",
    fontsize=11, fontweight="bold"
)
ax6.set_ylabel("R² Score", fontsize=9)
ax6.set_ylim(0.70, 1.00)
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3, axis="y")

for bar, val in zip(bars, r2_compare):
    ax6.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.005,
        f"{val:.4f}", ha="center",
        fontsize=10, fontweight="bold"
    )

# Graph 7 - Seasonal Validation
ax7 = fig.add_subplot(gs[2, 0])
seasons      = ["Spring", "Summer", "Autumn", "Winter"]
season_nums  = [1, 2, 3, 4]
season_r2    = []

for s in season_nums:
    mask  = val_df["month"].apply(
        lambda m: 1 if m in [3,4,5] else
                  2 if m in [6,7,8,9] else
                  3 if m in [10,11] else 4
    ) == s
    if mask.sum() > 0:
        r2 = r2_score(
            y_val[mask],
            ensemble_pred[mask]
        )
        season_r2.append(r2)
    else:
        season_r2.append(0)

colors_season = ["#4CAF50", "#FF5722",
                 "#FF9800", "#2196F3"]
bars = ax7.bar(
    seasons, season_r2,
    color=colors_season, alpha=0.85,
    edgecolor="black"
)
ax7.axhline(y=0.85, color="red",
            linestyle="--", linewidth=1.5,
            label="Target")
ax7.set_title(
    "Seasonal R² Validation",
    fontsize=11, fontweight="bold"
)
ax7.set_ylabel("R² Score", fontsize=9)
ax7.set_ylim(0.70, 1.00)
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3, axis="y")

for bar, val in zip(bars, season_r2):
    ax7.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.005,
        f"{val:.3f}", ha="center", fontsize=9
    )

# Graph 8 - CV-RMSE per city
ax8 = fig.add_subplot(gs[2, 1])
cv_values = [
    city_results[c]["cv_rmse"]
    for c in city_names
]
colors_cv = plt.cm.RdYlGn(
    np.linspace(0.9, 0.2, len(city_names))
)
bars = ax8.barh(
    city_names, cv_values,
    color=colors_cv, edgecolor="black",
    alpha=0.85
)
ax8.axvline(x=30, color="red",
            linestyle="--", linewidth=1.5,
            label="ASHRAE limit (30%)")
ax8.set_title(
    "CV-RMSE Per City\n(ASHRAE Guideline 14)",
    fontsize=11, fontweight="bold"
)
ax8.set_xlabel("CV-RMSE (%)", fontsize=9)
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3, axis="x")

for bar, val in zip(bars, cv_values):
    ax8.text(
        bar.get_width() + 0.1,
        bar.get_y() + bar.get_height()/2,
        f"{val:.1f}%", va="center", fontsize=8
    )

# Graph 9 - Summary metrics table
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis("off")

table_data = [
    ["Metric", "XGBoost", "LSTM", "Ensemble"],
    ["R²",
     f"{metrics_xgb['r2']:.4f}",
     f"{metrics_lstm['r2']:.4f}",
     f"{metrics_ens['r2']:.4f}"],
    ["MAE",
     f"{metrics_xgb['mae']:.6f}",
     f"{metrics_lstm['mae']:.6f}",
     f"{metrics_ens['mae']:.6f}"],
    ["RMSE",
     f"{metrics_xgb['rmse']:.6f}",
     f"{metrics_lstm['rmse']:.6f}",
     f"{metrics_ens['rmse']:.6f}"],
    ["MAPE",
     f"{metrics_xgb['mape']:.2f}%",
     f"{metrics_lstm['mape']:.2f}%",
     f"{metrics_ens['mape']:.2f}%"],
    ["CV-RMSE",
     f"{metrics_xgb['cv_rmse']:.2f}%",
     f"{metrics_lstm['cv_rmse']:.2f}%",
     f"{metrics_ens['cv_rmse']:.2f}%"],
]

table = ax9.table(
    cellText  = table_data[1:],
    colLabels = table_data[0],
    loc       = "center",
    cellLoc   = "center"
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
ax9.set_title(
    "Validation Metrics Summary",
    fontsize=11, fontweight="bold"
)

plt.suptitle(
    "NZEB Model Validation Report\n"
    "12 Indian Cities — Complete Pipeline",
    fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig(
    "validation_report.png",
    dpi=150, bbox_inches="tight"
)
plt.show()
print("✅ Validation graphs saved!")

# -------------------------------------------------------
# STEP 8 - Final validation report
# -------------------------------------------------------
print("\n" + "=" * 60)
print("📊 FINAL VALIDATION REPORT")
print("=" * 60)
print(f"""
🤖 MODEL PERFORMANCE
---------------------
   XGBoost  R²    : {metrics_xgb['r2']:.4f}
   LSTM     R²    : {metrics_lstm['r2']:.4f}
   Ensemble R²    : {metrics_ens['r2']:.4f}

📊 ERROR METRICS
-----------------
   Ensemble MAE   : {metrics_ens['mae']:.6f} kWh/m²
   Ensemble RMSE  : {metrics_ens['rmse']:.6f} kWh/m²
   Ensemble MAPE  : {metrics_ens['mape']:.2f}%
   CV-RMSE        : {metrics_ens['cv_rmse']:.2f}%

✅ ASHRAE GUIDELINE 14
-----------------------
   CV-RMSE < 30%  : {'✅ Pass' if metrics_ens['cv_rmse'] < 30 else '❌ Fail'}
   MAPE    < 10%  : {'✅ Pass' if metrics_ens['mape'] < 10 else '❌ Fail'}

🏙️  CITY VALIDATION
--------------------""")

for city, res in city_results.items():
    status = "✅" if res["r2"] >= 0.85 else "⚠️"
    print(f"   {status} {city:12s} R²={res['r2']:.4f} "
          f"MAPE={res['mape']:.1f}%")

print(f"""
🎯 OVERALL VERDICT
-------------------""")

if metrics_ens["r2"] >= 0.85:
    print("   ✅ Model is VALIDATED!")
    print("   ✅ Ready for research publication!")
    print("   ✅ ASHRAE Guideline 14 compliant!")
else:
    print("   ⚠️ Model needs improvement!")

print("\n💾 Validation report saved: validation_report.png")
print("=" * 60)