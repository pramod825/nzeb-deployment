# app.py - NZEB Flask Web Application
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
import json
import pvlib
import os
import datetime
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# -------------------------------------------------------
# Download RF model from Google Drive
# -------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def download_rf_model():
    """Download RF model from Google Drive if not exists"""
    rf_path1 = os.path.join(MODEL_DIR, "rf_model.pkl")
    rf_path2 = os.path.join(MODEL_DIR, "random_forest_model.pkl")

    if os.path.exists(rf_path1):
        print("✅ RF model already exists!")
        return rf_path1
    if os.path.exists(rf_path2):
        print("✅ RF model already exists!")
        return rf_path2

    print("⬇️  Downloading RF model from Google Drive...")
    import gdown
    file_id = "1kfbtbv2VVou-XjOq7bVq8-GJRVS_BmEF"
    url     = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, rf_path1, quiet=False)
    print("✅ RF model downloaded successfully!")
    return rf_path1

# -------------------------------------------------------
# ✅ NEW: Load tiny JSON files at startup (safe, small)
# -------------------------------------------------------
with open(os.path.join(MODEL_DIR, "cities.json"), "r") as f:
    cities = json.load(f)
print(f"✅ Cities loaded! ({len(cities)} cities)")

with open(os.path.join(MODEL_DIR, "building.json"), "r") as f:
    building = json.load(f)
print("✅ Building params loaded!")

with open(os.path.join(MODEL_DIR, "city_encoding.json"), "r") as f:
    city_encoding = json.load(f)
print("✅ City encoding loaded!")

# -------------------------------------------------------
# ✅ NEW: Lazy model loader — loads only on first request
# -------------------------------------------------------
_models = {}

def get_models():
    if _models:
        return _models  # already loaded, return immediately

    print("🚀 Loading models on first request...")

    # Download & load RF model
    rf_path = download_rf_model()
    with open(rf_path, "rb") as f:
        _models["rf"] = pickle.load(f)
    print("✅ RF model loaded!")

    # Load XGBoost model
    with open(os.path.join(MODEL_DIR, "xgb_model.pkl"), "rb") as f:
        _models["xgb"] = pickle.load(f)
    print("✅ XGBoost model loaded!")

    # Load Scaler
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        _models["scaler"] = pickle.load(f)
    print("✅ Scaler loaded!")

    # Load Feature columns
    with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb") as f:
        _models["feature_cols"] = pickle.load(f)
    print(f"✅ Feature cols loaded!")

    # Load LSTM model
   # DELETE these lines
    from tensorflow.keras.models import load_model  # type: ignore
    lstm_model = load_model(
    os.path.join(MODEL_DIR, "lstm_model.keras")
)
    print("✅ LSTM model loaded!")

    print("🎉 All models loaded successfully!")
    return _models


# -------------------------------------------------------
# Helper function - Calculate all features (UNCHANGED)
# -------------------------------------------------------
def calculate_features(
    city, month, day, hour,
    temperature, humidity,
    solar_radiation, wind_speed,
    cloud_cover, pressure=101.325
):
    info = cities[city]

    site = pvlib.location.Location(
        latitude  = info["lat"],
        longitude = info["lon"],
        altitude  = info["altitude"],
        tz        = info["timezone"],
        name      = city
    )

    dt = pd.Timestamp(
        year=2024, month=month,
        day=day,   hour=hour,
        tz=info["timezone"]
    )

    solar_pos  = site.get_solarposition(pd.DatetimeIndex([dt]))
    solar_alt  = float(solar_pos["apparent_elevation"].iloc[0])
    solar_az   = float(solar_pos["azimuth"].iloc[0])
    solar_zen  = float(solar_pos["apparent_zenith"].iloc[0])
    is_daytime = 1 if solar_alt > 0 else 0

    ghi  = float(solar_radiation)
    disc = pvlib.irradiance.disc(
        np.array([ghi]),
        np.array([solar_zen]),
        pd.DatetimeIndex([dt])
    )
    dni = max(0.0, float(np.array(disc["dni"]).flatten()[0]))
    dhi = max(0.0, ghi - dni * np.cos(np.radians(solar_zen)))

    walls = {
        "north": {"tilt": 90, "azimuth": 0},
        "south": {"tilt": 90, "azimuth": 180},
        "east":  {"tilt": 90, "azimuth": 90},
        "west":  {"tilt": 90, "azimuth": 270},
    }

    facade_radiation = {}
    for wall_name, angles in walls.items():
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
        val = float(np.array(poa["poa_global"]).flatten()[0])
        facade_radiation[wall_name] = max(0.0, float(np.nan_to_num(val)))

    eff = float(building["bipv_efficiency"])
    pr  = float(building["bipv_pr"])

    bipv_south_w = facade_radiation["south"] * float(building["bipv_area_south"]) * eff * pr * is_daytime
    bipv_east_w  = facade_radiation["east"]  * float(building["bipv_area_east"])  * eff * pr * is_daytime
    bipv_west_w  = facade_radiation["west"]  * float(building["bipv_area_west"])  * eff * pr * is_daytime
    bipv_total_w   = bipv_south_w + bipv_east_w + bipv_west_w
    bipv_south_kwh = bipv_south_w / 1000
    bipv_east_kwh  = bipv_east_w  / 1000
    bipv_west_kwh  = bipv_west_w  / 1000
    bipv_total_kwh = bipv_total_w / 1000
    bipv_m2        = bipv_total_kwh / float(building["total_area"])

    date_obj    = datetime.date(2024, month, day)
    day_of_week = date_obj.weekday()
    day_of_year = date_obj.timetuple().tm_yday
    is_weekend  = 1 if day_of_week >= 5 else 0
    is_peak     = 1 if 9 <= hour <= 18 else 0

    cdd        = max(0.0, temperature - 24)
    hdd        = max(0.0, 18 - temperature)
    discomfort = (0.4 * (temperature + 0.99 * humidity) + 4.9)
    heat_idx   = (temperature + 0.33 * (humidity / 100 * 6.105 *
                  np.exp(17.27 * temperature / (237.7 + temperature))) - 4.0)

    total_facade = sum(facade_radiation.values())
    rad_ratio    = (total_facade / solar_radiation if solar_radiation > 0 else 0)

    solar_heat_gain = (
        float(building["shgc_south"]) *
        (float(building["glass_area_north"]) + float(building["glass_area_south"]) +
         float(building["glass_area_east"])  + float(building["glass_area_west"])) *
        solar_radiation / float(building["total_area"])
    )

    envelope_loss = (
        float(building["u_wall"]) *
        (float(building["wall_area_north"]) + float(building["wall_area_south"]) +
         float(building["wall_area_east"])  + float(building["wall_area_west"])) *
        abs(temperature - 22) / float(building["total_area"])
    )

    thermal_comfort = 1 if (20 <= temperature <= 26 and 30 <= humidity <= 60) else 0

    season = (
        1 if month in [3,4,5]   else
        2 if month in [6,7,8,9] else
        3 if month in [10,11]   else 4
    )

    city_encoded = city_encoding[city]
    climate_zone = info["climate_zone"]

    input_data = {
        "temperature":               temperature,
        "relative_humidity":         humidity,
        "solar_radiation":           solar_radiation,
        "wind_speed":                wind_speed,
        "cloud_cover":               cloud_cover,
        "pressure":                  pressure,
        "year":                      2024,
        "month":                     month,
        "day":                       day,
        "hour":                      hour,
        "day_of_week":               day_of_week,
        "is_weekend":                is_weekend,
        "climate_zone":              climate_zone,
        "altitude":                  info["altitude"],
        "latitude":                  info["lat"],
        "longitude":                 info["lon"],
        "solar_altitude":            solar_alt,
        "solar_azimuth":             solar_az,
        "solar_zenith":              solar_zen,
        "is_daytime":                is_daytime,
        "hour_angle":                (hour - 12) * 15,
        "radiation_north":           facade_radiation["north"],
        "radiation_south":           facade_radiation["south"],
        "radiation_east":            facade_radiation["east"],
        "radiation_west":            facade_radiation["west"],
        "bipv_south_w":              bipv_south_w,
        "bipv_east_w":               bipv_east_w,
        "bipv_west_w":               bipv_west_w,
        "bipv_total_w":              bipv_total_w,
        "bipv_south_kwh":            bipv_south_kwh,
        "bipv_east_kwh":             bipv_east_kwh,
        "bipv_west_kwh":             bipv_west_kwh,
        "bipv_total_kwh":            bipv_total_kwh,
        "bipv_kwh_m2":               bipv_m2,
        "city_encoded":              city_encoded,
        "sin_hour":                  np.sin(2*np.pi*hour/24),
        "cos_hour":                  np.cos(2*np.pi*hour/24),
        "sin_month":                 np.sin(2*np.pi*month/12),
        "cos_month":                 np.cos(2*np.pi*month/12),
        "sin_doy":                   np.sin(2*np.pi*day_of_year/365),
        "cos_doy":                   np.cos(2*np.pi*day_of_year/365),
        "is_peak_hour":              is_peak,
        "CDD":                       cdd,
        "HDD":                       hdd,
        "discomfort_index":          discomfort,
        "heat_index":                heat_idx,
        "total_facade_radiation":    total_facade,
        "radiation_ratio":           rad_ratio,
        "thermal_comfort":           thermal_comfort,
        "solar_heat_gain":           solar_heat_gain,
        "envelope_loss":             envelope_loss,
        "season":                    season,
        "temp_solar_interaction":    temperature * solar_radiation / 1000,
        "humidity_temp_interaction": humidity * temperature / 100,
        "cdd_solar_interaction":     cdd * solar_radiation / 1000,
    }

    return input_data, facade_radiation, bipv_m2


# -------------------------------------------------------
# Routes
# -------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html", cities=cities)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ✅ NEW: Get models lazily (loads on first request only)
        models       = get_models()
        rf_model     = models["rf"]
        xgb_model    = models["xgb"]
        lstm_model   = models["lstm"]
        scaler       = models["scaler"]
        feature_cols = models["feature_cols"]

        data = request.json

        city            = data["city"]
        month           = int(data["month"])
        day             = int(data["day"])
        hour            = int(data["hour"])
        temperature     = float(data["temperature"])
        humidity        = float(data["humidity"])
        solar_radiation = float(data["solar_radiation"])
        wind_speed      = float(data["wind_speed"])
        cloud_cover     = float(data["cloud_cover"])

        input_data, facade_rad, bipv_m2 = calculate_features(
            city, month, day, hour,
            temperature, humidity,
            solar_radiation, wind_speed,
            cloud_cover
        )

        input_df     = pd.DataFrame([input_data])
        input_df     = input_df[feature_cols]
        input_scaled = scaler.transform(input_df)

        rf_pred  = float(rf_model.predict(input_scaled)[0])
        xgb_pred = float(xgb_model.predict(input_scaled)[0])
        lstm_in  = input_scaled.reshape(1, 1, input_scaled.shape[1])
        lstm_pred = float(lstm_model.predict(lstm_in, verbose=0)[0][0])

        ensemble   = (rf_pred + xgb_pred + lstm_pred) / 3
        net_energy = ensemble - bipv_m2

        if ensemble < 0.05:
            rating = "Very Low Energy (Excellent NZEB)"
        elif ensemble < 0.10:
            rating = "Low Energy (Good NZEB)"
        elif ensemble < 0.15:
            rating = "Moderate Energy"
        elif ensemble < 0.20:
            rating = "High Energy"
        else:
            rating = "Very High Energy (Poor NZEB)"

        nzeb_status = (
            "Net Zero Achieved! ✅"
            if net_energy <= 0
            else f"Net Zero Gap: {net_energy:.6f} kWh/m²"
        )

        return jsonify({
            "success":       True,
            "city":          city,
            "rf_pred":       round(rf_pred,    6),
            "xgb_pred":      round(xgb_pred,   6),
            "lstm_pred":     round(lstm_pred,   6),
            "ensemble":      round(ensemble,    6),
            "bipv_kwh_m2":   round(bipv_m2,     6),
            "net_energy":    round(net_energy,   6),
            "energy_rating": rating,
            "nzeb_status":   nzeb_status,
            "facade_north":  round(facade_rad["north"], 2),
            "facade_south":  round(facade_rad["south"], 2),
            "facade_east":   round(facade_rad["east"],  2),
            "facade_west":   round(facade_rad["west"],  2),
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/health")
def health():
    return jsonify({
        "status":   "healthy",
        "models":   "loaded" if _models else "not loaded yet",
        "cities":   len(cities),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)