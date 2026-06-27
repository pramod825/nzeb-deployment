# app.py - NZEB Flask Web Application (Updated)
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

app = Flask(__name__, template_folder="templates")

# -------------------------------------------------------
# Download model from Google Drive if not present
# -------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RF_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
RF_MODEL_GDRIVE_ID = "1UVuZd1qxsksb0QAxKAI-g-PlazbSXWyX"

if not os.path.exists(RF_MODEL_PATH):
    print("⬇️  Downloading RF model from Google Drive...")
    try:
        import gdown
        gdown.download(
            f"https://drive.google.com/uc?id={RF_MODEL_GDRIVE_ID}",
            RF_MODEL_PATH,
            quiet=False
        )
        print("✅ RF model downloaded successfully!")
    except Exception as e:
        print(f"❌ Failed to download RF model: {e}")
        raise RuntimeError(f"Could not download random_forest_model.pkl: {e}")

# -------------------------------------------------------
# Load all models and files at startup
# -------------------------------------------------------
print("🚀 Loading models...")

with open(RF_MODEL_PATH, "rb") as f:
    rf_model = pickle.load(f)
print("✅ RF model loaded!")

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
print("✅ Scaler loaded!")

with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb") as f:
    feature_cols = pickle.load(f)
print(f"✅ Feature cols loaded! ({len(feature_cols)} features)")

with open(os.path.join(MODEL_DIR, "cities.json"), "r") as f:
    cities = json.load(f)
print(f"✅ Cities loaded! ({len(cities)} cities)")

with open(os.path.join(MODEL_DIR, "building.json"), "r") as f:
    building = json.load(f)
print("✅ Building params loaded!")

with open(os.path.join(MODEL_DIR, "city_encoding.json"), "r") as f:
    city_encoding = json.load(f)
print("✅ City encoding loaded!")

print("🎉 All models loaded successfully!")

# -------------------------------------------------------
# India climate zone lookup by lat/lon
# -------------------------------------------------------
def get_climate_zone(lat, lon):
    """Simple climate zone estimation for Indian cities"""
    if lat > 28:
        return 1  # Cold/Composite (North India)
    elif lat > 23:
        return 2  # Composite (Central India)
    elif lon > 80 and lat < 20:
        return 3  # Hot & Humid (Coastal/South-East)
    elif lon < 75:
        return 4  # Hot & Dry (West India / Rajasthan)
    else:
        return 2  # Default Composite

def get_city_info(city_name, lat, lon):
    """Get city info — use known cities or estimate for new ones"""
    # Check if city exists in our database
    for known_city, info in cities.items():
        if known_city.lower() == city_name.lower():
            return known_city, info, known_city

    # For unknown cities, estimate based on lat/lon
    climate_zone = get_climate_zone(lat, lon)

    # Find nearest known city encoding
    nearest_city = min(cities.keys(), key=lambda c: (
        (cities[c]["lat"] - lat)**2 + (cities[c]["lon"] - lon)**2
    ))
    city_enc = city_encoding.get(nearest_city, 0)

    # Estimate altitude from lat (rough India estimate)
    if lat > 25 and lon < 77:
        altitude = 300  # North-West plains
    elif lat > 20 and lon > 85:
        altitude = 50   # Coastal East
    else:
        altitude = 200  # Default

    estimated_info = {
        "lat": lat,
        "lon": lon,
        "altitude": altitude,
        "timezone": "Asia/Kolkata",
        "climate_zone": climate_zone,
        "state": "India",
        "zone_name": ["", "Cold/Composite", "Composite", "Hot & Humid", "Hot & Dry"][climate_zone]
    }
    return city_name, estimated_info, nearest_city

# -------------------------------------------------------
# Feature calculation
# -------------------------------------------------------
def calculate_features(city_name, lat, lon, month, day, hour,
                        temperature, humidity, solar_radiation,
                        wind_speed, cloud_cover, pressure=101.325):

    resolved_name, info, nearest_city = get_city_info(city_name, lat, lon)

    site = pvlib.location.Location(
        latitude=info["lat"], longitude=info["lon"],
        altitude=info["altitude"], tz=info["timezone"], name=city_name
    )

    dt = pd.Timestamp(year=2024, month=month, day=day, hour=hour, tz=info["timezone"])
    solar_pos = site.get_solarposition(pd.DatetimeIndex([dt]))
    solar_alt = float(solar_pos["apparent_elevation"].iloc[0])
    solar_az  = float(solar_pos["azimuth"].iloc[0])
    solar_zen = float(solar_pos["apparent_zenith"].iloc[0])
    is_daytime = 1 if solar_alt > 0 else 0

    ghi  = float(solar_radiation)
    disc = pvlib.irradiance.disc(np.array([ghi]), np.array([solar_zen]), pd.DatetimeIndex([dt]))
    dni  = max(0.0, float(np.array(disc["dni"]).flatten()[0]))
    dhi  = max(0.0, ghi - dni * np.cos(np.radians(solar_zen)))

    walls = {
        "north": {"tilt": 90, "azimuth": 0},
        "south": {"tilt": 90, "azimuth": 180},
        "east":  {"tilt": 90, "azimuth": 90},
        "west":  {"tilt": 90, "azimuth": 270},
    }
    facade_radiation = {}
    for wall_name, angles in walls.items():
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=angles["tilt"], surface_azimuth=angles["azimuth"],
            solar_zenith=np.array([solar_zen]), solar_azimuth=np.array([solar_az]),
            dni=np.array([dni]), ghi=np.array([ghi]), dhi=np.array([dhi]), model="isotropic"
        )
        val = float(np.array(poa["poa_global"]).flatten()[0])
        facade_radiation[wall_name] = max(0.0, float(np.nan_to_num(val)))

    eff = float(building["bipv_efficiency"])
    pr  = float(building["bipv_pr"])
    bipv_south_w = facade_radiation["south"] * float(building["bipv_area_south"]) * eff * pr * is_daytime
    bipv_east_w  = facade_radiation["east"]  * float(building["bipv_area_east"])  * eff * pr * is_daytime
    bipv_west_w  = facade_radiation["west"]  * float(building["bipv_area_west"])  * eff * pr * is_daytime
    bipv_total_w = bipv_south_w + bipv_east_w + bipv_west_w
    bipv_m2      = (bipv_total_w / 1000) / float(building["total_area"])

    date_obj    = datetime.date(2024, month, day)
    day_of_week = date_obj.weekday()
    day_of_year = date_obj.timetuple().tm_yday
    is_weekend  = 1 if day_of_week >= 5 else 0
    is_peak     = 1 if 9 <= hour <= 18 else 0

    cdd        = max(0.0, temperature - 24)
    hdd        = max(0.0, 18 - temperature)
    discomfort = 0.4 * (temperature + 0.99 * humidity) + 4.9
    heat_idx   = temperature + 0.33 * (humidity / 100 * 6.105 * np.exp(17.27 * temperature / (237.7 + temperature))) - 4.0

    total_facade    = sum(facade_radiation.values())
    rad_ratio       = total_facade / solar_radiation if solar_radiation > 0 else 0
    solar_heat_gain = (float(building["shgc_south"]) *
                       (float(building["glass_area_north"]) + float(building["glass_area_south"]) +
                        float(building["glass_area_east"])  + float(building["glass_area_west"])) *
                       solar_radiation / float(building["total_area"]))
    envelope_loss   = (float(building["u_wall"]) *
                       (float(building["wall_area_north"]) + float(building["wall_area_south"]) +
                        float(building["wall_area_east"])  + float(building["wall_area_west"])) *
                       abs(temperature - 22) / float(building["total_area"]))
    thermal_comfort = 1 if (20 <= temperature <= 26 and 30 <= humidity <= 60) else 0
    season = (1 if month in [3,4,5] else 2 if month in [6,7,8,9] else 3 if month in [10,11] else 4)

    city_encoded = city_encoding.get(nearest_city, 0)
    climate_zone = info["climate_zone"]

    input_data = {
        "temperature": temperature, "relative_humidity": humidity,
        "solar_radiation": solar_radiation, "wind_speed": wind_speed,
        "cloud_cover": cloud_cover, "pressure": pressure,
        "year": 2024, "month": month, "day": day, "hour": hour,
        "day_of_week": day_of_week, "is_weekend": is_weekend,
        "climate_zone": climate_zone, "altitude": info["altitude"],
        "latitude": info["lat"], "longitude": info["lon"],
        "solar_altitude": solar_alt, "solar_azimuth": solar_az,
        "solar_zenith": solar_zen, "is_daytime": is_daytime,
        "hour_angle": (hour - 12) * 15,
        "radiation_north": facade_radiation["north"],
        "radiation_south": facade_radiation["south"],
        "radiation_east":  facade_radiation["east"],
        "radiation_west":  facade_radiation["west"],
        "bipv_south_w": bipv_south_w, "bipv_east_w": bipv_east_w,
        "bipv_west_w": bipv_west_w,   "bipv_total_w": bipv_total_w,
        "bipv_south_kwh": bipv_south_w/1000, "bipv_east_kwh": bipv_east_w/1000,
        "bipv_west_kwh":  bipv_west_w/1000,  "bipv_total_kwh": bipv_total_w/1000,
        "bipv_kwh_m2": bipv_m2, "city_encoded": city_encoded,
        "sin_hour": np.sin(2*np.pi*hour/24), "cos_hour": np.cos(2*np.pi*hour/24),
        "sin_month": np.sin(2*np.pi*month/12), "cos_month": np.cos(2*np.pi*month/12),
        "sin_doy": np.sin(2*np.pi*day_of_year/365), "cos_doy": np.cos(2*np.pi*day_of_year/365),
        "is_peak_hour": is_peak, "CDD": cdd, "HDD": hdd,
        "discomfort_index": discomfort, "heat_index": heat_idx,
        "total_facade_radiation": total_facade, "radiation_ratio": rad_ratio,
        "thermal_comfort": thermal_comfort, "solar_heat_gain": solar_heat_gain,
        "envelope_loss": envelope_loss, "season": season,
        "temp_solar_interaction": temperature * solar_radiation / 1000,
        "humidity_temp_interaction": humidity * temperature / 100,
        "cdd_solar_interaction": cdd * solar_radiation / 1000,
    }
    return input_data, facade_radiation, bipv_m2, info

# -------------------------------------------------------
# Monthly chart data generator
# -------------------------------------------------------
def get_monthly_data(city_name, lat, lon, temperature, humidity,
                     solar_radiation, wind_speed, cloud_cover):
    months = range(1, 13)
    energy_vals = []
    bipv_vals   = []

    # Seasonal temp/solar adjustments for realism
    temp_adj   = [0,-2,-1,2,6,4,2,1,0,-1,-2,-1]
    solar_adj  = [0.7,0.8,0.9,1.0,1.0,0.8,0.7,0.7,0.85,0.9,0.8,0.7]

    for i, m in enumerate(months):
        try:
            t  = temperature + temp_adj[i]
            sr = solar_radiation * solar_adj[i]
            input_data, _, bm2, _ = calculate_features(
                city_name, lat, lon, m, 15, 14, t, humidity, sr, wind_speed, cloud_cover
            )
            df     = pd.DataFrame([input_data])[feature_cols]
            scaled = scaler.transform(df)
            pred   = float(rf_model.predict(scaled)[0])
            energy_vals.append(round(pred, 4))
            bipv_vals.append(round(bm2, 4))
        except:
            energy_vals.append(0)
            bipv_vals.append(0)

    return energy_vals, bipv_vals

# -------------------------------------------------------
# Routes
# -------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html", cities=cities)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data            = request.json
        city_name       = data["city"]
        lat             = float(data.get("lat", 26.9))
        lon             = float(data.get("lon", 75.8))
        month           = int(data["month"])
        day             = int(data["day"])
        hour            = int(data["hour"])
        temperature     = float(data["temperature"])
        humidity        = float(data["humidity"])
        solar_radiation = float(data["solar_radiation"])
        wind_speed      = float(data["wind_speed"])
        cloud_cover     = float(data["cloud_cover"])

        input_data, facade_rad, bipv_m2, info = calculate_features(
            city_name, lat, lon, month, day, hour,
            temperature, humidity, solar_radiation, wind_speed, cloud_cover
        )

        df           = pd.DataFrame([input_data])[feature_cols]
        input_scaled = scaler.transform(df)
        rf_pred      = float(rf_model.predict(input_scaled)[0])
        net_energy   = rf_pred - bipv_m2

        # NZEB status & recommendation
        nzeb_pct = min(100, max(0, (bipv_m2 / rf_pred * 100) if rf_pred > 0 else 100))
        if net_energy <= 0:
            recommendation = "✅ Net Zero Achieved! Your building generates enough solar energy to cover its consumption. Maintain current BIPV system and insulation."
            status_color   = "green"
        elif nzeb_pct >= 75:
            recommendation = "🟡 Almost there! You are 75%+ toward Net Zero. Add 25% more BIPV panels on South/West facade to achieve Net Zero."
            status_color   = "yellow"
        elif nzeb_pct >= 50:
            recommendation = "🟠 Halfway to Net Zero. Increase BIPV coverage, improve window glazing (low SHGC glass), and add roof insulation."
            status_color   = "orange"
        else:
            recommendation = "🔴 High energy gap. Major improvements needed: expand BIPV system, upgrade HVAC efficiency, improve building envelope, and use smart controls."
            status_color   = "red"

        # Energy rating
        if rf_pred < 0.05:
            rating = "⭐⭐⭐⭐⭐ Excellent (Very Low Energy)"
        elif rf_pred < 0.10:
            rating = "⭐⭐⭐⭐ Good (Low Energy)"
        elif rf_pred < 0.15:
            rating = "⭐⭐⭐ Moderate Energy"
        elif rf_pred < 0.20:
            rating = "⭐⭐ High Energy"
        else:
            rating = "⭐ Very High Energy"

        # Monthly chart data
        monthly_energy, monthly_bipv = get_monthly_data(
            city_name, lat, lon, temperature, humidity,
            solar_radiation, wind_speed, cloud_cover
        )

        return jsonify({
            "success":        True,
            "city":           city_name,
            "rf_pred":        round(rf_pred, 6),
            "bipv_kwh_m2":    round(bipv_m2, 6),
            "net_energy":     round(net_energy, 6),
            "nzeb_pct":       round(nzeb_pct, 1),
            "energy_rating":  rating,
            "recommendation": recommendation,
            "status_color":   status_color,
            "facade_north":   round(facade_rad["north"], 2),
            "facade_south":   round(facade_rad["south"], 2),
            "facade_east":    round(facade_rad["east"],  2),
            "facade_west":    round(facade_rad["west"],  2),
            "climate_zone":   info["zone_name"] if isinstance(info.get("zone_name"), str) else str(info.get("climate_zone", "")),
            "monthly_energy": monthly_energy,
            "monthly_bipv":   monthly_bipv,
        })

    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})


@app.route("/geocode", methods=["POST"])
def geocode():
    """Get lat/lon for any Indian city using a simple lookup or fallback"""
    try:
        city_name = request.json.get("city", "").strip()

        # Check known cities first
        for known, info in cities.items():
            if known.lower() == city_name.lower():
                return jsonify({
                    "success": True,
                    "lat": info["lat"], "lon": info["lon"],
                    "found": True, "matched": known,
                    "zone": info.get("zone_name", ""),
                    "state": info.get("state", "")
                })

        # Popular Indian cities fallback
        indian_cities = {
            "jodhpur":      (26.2389, 73.0243), "jaipur":     (26.9124, 75.7873),
            "udaipur":      (24.5854, 73.7125), "kota":       (25.2138, 75.8648),
            "ajmer":        (26.4499, 74.6399), "bikaner":    (28.0229, 73.3119),
            "agra":         (27.1767, 78.0081), "lucknow":    (26.8467, 80.9462),
            "kanpur":       (26.4499, 80.3319), "varanasi":   (25.3176, 82.9739),
            "patna":        (25.5941, 85.1376), "bhopal":     (23.2599, 77.4126),
            "indore":       (22.7196, 75.8577), "nagpur":     (21.1458, 79.0882),
            "surat":        (21.1702, 72.8311), "vadodara":   (22.3072, 73.1812),
            "rajkot":       (22.3039, 70.8022), "amritsar":   (31.6340, 74.8723),
            "ludhiana":     (30.9010, 75.8573), "chandigarh": (30.7333, 76.7794),
            "dehradun":     (30.3165, 78.0322), "shimla":     (31.1048, 77.1734),
            "coimbatore":   (11.0168, 76.9558), "madurai":    (9.9252,  78.1198),
            "visakhapatnam":(17.6868, 83.2185), "vijayawada": (16.5062, 80.6480),
            "mysuru":       (12.2958, 76.6394), "hubli":      (15.3647, 75.1240),
            "thiruvananthapuram":(8.5241,76.9366),"kochi":    (9.9312,  76.2673),
            "bhubaneswar":  (20.2961, 85.8245), "raipur":     (21.2514, 81.6296),
            "ranchi":       (23.3441, 85.3096), "guwahati":   (26.1445, 91.7362),
            "shillong":     (25.5788, 91.8933), "imphal":     (24.8170, 93.9368),
            "jammu":        (32.7266, 74.8570), "srinagar":   (34.0837, 74.7973),
        }

        key = city_name.lower()
        if key in indian_cities:
            lat, lon = indian_cities[key]
            return jsonify({"success": True, "lat": lat, "lon": lon, "found": True})

        # Default fallback — center of India
        return jsonify({
            "success": True, "lat": 22.0, "lon": 78.0,
            "found": False,
            "message": f"City '{city_name}' not in database. Using estimated location."
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "models": "loaded", "cities": len(cities), "features": len(feature_cols)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)