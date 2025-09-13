# WEATHER-PREDICTION
This project predicts tomorrow’s mean temperature for selected Indian cities using a combination of historical weather data and a Polynomial Regression model. It also compares the prediction against the official Open-Meteo 7-day forecast.

# Features
🔎 Geocoding: Automatically fetches latitude/longitude and timezone of a city.
📊 Historical Data: Pulls the last 120 days of max/min temperature data from the Open-Meteo Archive API.
🤖 Machine Learning: Fits a polynomial regression model (degree 3) on historical mean temperature trends.
📅 Forecast Comparison: Fetches the Open-Meteo 7-day forecast for validation.
🖥️ CLI-based Interface: Simple menu to select from popular Indian cities (Kolkata, Delhi, Mumbai, Chennai, Bengaluru, Hyderabad).

# How it works??
User selects a city from the menu.
Program geocodes the city → gets latitude/longitude/timezone.
Historical weather data (last 120 days, excluding last 3 days) is fetched.
Mean daily temperature is calculated and used to train a polynomial regression model.
Model predicts tomorrow’s mean temperature.
Open-Meteo forecast for tomorrow is fetched and compared with model output.

