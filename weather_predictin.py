from datetime import date, timedelta
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

HIST_URL = "https://archive-api.open-meteo.com/v1/archive"


FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def geocode_city(name):
 
    params = {"name": name, "count": 1, "language": "en", "format": "json"}

    r = requests.get(GEOCODE_URL, params=params, timeout=30)

    r.raise_for_status()
    # data = r.json()-> The server replies in JSON text,.json() converts it into a Python dictionary/list so you can use it directly
    data = r.json()
    #print(data)
    if not data.get("results"):
        # Defensive: ensure callers get a clean error message if no match
        raise ValueError(f"Could not geocode city '{name}'. Try a different spelling.")
    res = data["results"][0]
    return {
        "name": res.get("name"),
        "latitude": res["latitude"],
        "longitude": res["longitude"],
        "timezone": res.get("timezone", "auto"),
        "country": res.get("country"),
        "admin1": res.get("admin1"),
    }


def fetch_history(lat, lon, start_date, end_date, timezone="auto"):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        # Converts Python date objects into API-friendly strings (e.g., "2025-06-01")
        "end_date": end_date.isoformat(),
        "daily": ["temperature_2m_max", "temperature_2m_min"],  # Asks specifically for daily max & min temperatures
        "timezone": timezone,
    }
    r = requests.get(HIST_URL, params=params, timeout=60)


    r.raise_for_status()  # immediately raise an error if response code is not 200
    d = r.json()  # parse the returned JSON into a Python dictionary.
    #print(d)
    daily = d.get("daily", {})  # Extract the "daily" key from the dictionary
    # If "daily" is missing or doesn’t contain "time", abort with a clear error
    if not daily or "time" not in daily:
        # Happens if API has a gap or the dates are out of range
        raise RuntimeError("Historical data not available for the requested range.")

    # Turns the JSON (lists of values like ["2025-06-01", "2025-06-02", ...]) into a pandas DataFrame
    # Columns include "time", "temperature_2m_max", "temperature_2m_min"
    df = pd.DataFrame(daily)

    # Coerce numeric columns; any invalid string/None becomes NaN
    # Ensures data is numeric (float).If the API gives weird values (e.g., "null"), they become NaN
    for col in ["temperature_2m_max", "temperature_2m_min"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Target variable: simple average of daily max/min
    df["temp_mean"] = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2.0
    df["date"] = pd.to_datetime(df["time"])

    # Drop rows where the target is missing to keep the regression clean
    before = len(df)
    df = df.dropna(subset=["temp_mean"]).reset_index(drop=True)
    after = len(df)
    if after < before:
        print(f"[info] Dropped {before - after} rows with NaN temperatures")

    return df[["date", "temperature_2m_min", "temperature_2m_max", "temp_mean"]]


def fetch_forecast(lat, lon, timezone="auto"):
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "temperature_2m_min"],
        "forecast_days": 7,
        "timezone": timezone,
    }
    r = requests.get(FORECAST_URL, params=params, timeout=30)
    r.raise_for_status()
    d = r.json()
    daily = d.get("daily", {})
    if not daily or "time" not in daily:
        # Return an empty DF if the API doesn’t have forecast data
        return pd.DataFrame(columns=["date", "temperature_2m_min", "temperature_2m_max", "temp_mean"])
    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"])
    df["temperature_2m_max"] = pd.to_numeric(df["temperature_2m_max"], errors="coerce")
    df["temperature_2m_min"] = pd.to_numeric(df["temperature_2m_min"], errors="coerce")
    df["temp_mean"] = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2.0
    return df[["date", "temperature_2m_min", "temperature_2m_max", "temp_mean"]]


def build_xy(df):
 
    df = df.sort_values("date").reset_index(drop=True)
    start = df["date"].min()  # Gets the earliest date in the DataFrame
 
    df["x"] = (df["date"] - start).dt.days.astype(int)

    X = df[["x"]].values
    y = df["temp_mean"].values.astype(float)
    return df, X, y, start


def fit_poly_regression(X, y, degree=3):
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("linreg", LinearRegression())
    ])
    model.fit(X, y)
    return model


def main():
    # Step 1: Simple city menu (you can add more cities here)
    cities = ["Kolkata", "Delhi", "Mumbai", "Chennai", "Bengaluru", "Hyderabad"]
    print("Choose a city:")
    for i, c in enumerate(cities, 1):
        print(f"{i}. {c}")

    # Get selection (defaults to 1 if input is empty/invalid)
    try:
        choice = int(input(f"Enter number between 1 and {len(cities)}:- "))
        if not (1 <= choice <= len(cities)):
            raise ValueError
    except Exception:
        print("[info] Invalid choice, defaulting to 1.")
        choice = 1

    city = cities[choice - 1]

    # Step 2: Geocode -> (lat, lon, tz)
    place = geocode_city(city)
    #print(place)
    lat, lon, tz = place["latitude"], place["longitude"], place["timezone"]

    # Step 3: Fetch history for last 120 days (skip last 3 days to avoid incomplete archive)
    today = date.today()
    start = today - timedelta(days=120)
    hist_end = today - timedelta(days=3)
    hist_df = fetch_history(lat, lon, start, hist_end, tz)
    #print(hist_df)

    if hist_df.empty or len(hist_df) < 5:
        print(f"Not enough historical samples for {city}. Try another.")
        return

    # Prepare X, y for regression
    hist_df, X, y, base_date = build_xy(hist_df)
    #print(hist_df, X, y, base_date)
 
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
 
    hist_df = hist_df.loc[mask].reset_index(drop=True)
    # above line keeps only the valid rows of your historical weather data and cleans up the row numbers, so later code doesn’t get confused by gaps in the index.

    if len(y) < 5:
        print(f"Too few clean samples to fit model for {city}.")
        return

    # Step 4: Fit regression model (degree = 3, safe cap)
    degree = 3
    model = fit_poly_regression(X, y, degree=degree)

    # Step 5: Predict tomorrow
    tomorrow = today + timedelta(days=1)
 
    x_tomorrow = np.array([[(pd.Timestamp(tomorrow) - base_date).days]])  # [[103]]
    y_pred = float(model.predict(x_tomorrow)[0])  # temprerature of tomorrow(predicted)

    # Step 6: Fetch forecast for comparison

    fc_df = fetch_forecast(lat, lon, tz)
    #print(fc_df)
    if not fc_df.empty:
        tomorrow_data = fc_df.loc[fc_df["date"].dt.date == tomorrow, "temp_mean"]
        #print(tomorrow_data)
        if not tomorrow_data.empty:
            fc_val = float(tomorrow_data.iloc[0])  # gets the first value from the Series

    # Step 7: Display results
    parts = []
    if place.get("name"):
        parts.append(place["name"])
    if place.get("admin1"):
        parts.append(place["admin1"])
    if place.get("country"):
        parts.append(place["country"])
    loc_str = ", ".join(parts)
    print(f"\n📍 Location: {loc_str}")
    print(f"📅 Tomorrow ({tomorrow.isoformat()})")
    print(f"🔹 Model predicted mean temp: {y_pred:.2f} °C")

    if fc_val is not None:
        diff = y_pred - fc_val
        print(f"🔹 Open-Meteo forecast: {fc_val:.2f} °C")
        print(f"🔹 Difference (model - forecast): {diff:+.2f} °C")
    else:
        print("⚠️ Open-Meteo forecast unavailable for comparison.")


if __name__ == "__main__":
    # Standard Python entry-point guard; allows importing this module without executing main()
    main()