# %%
from datetime import datetime, timedelta
import pytz
import csv
import os
import sys
import requests

tz = pytz.timezone("America/Los_Angeles")
now = datetime.now(tz)

day = now.weekday()   # 0=Mon, 6=Sun
hour = now.hour
minute = now.minute



# %%
TOKEN = os.getenv("DENSITY_TOKEN")
print(TOKEN)


# %%
url = 'https://api.density.io/v2/spaces?page=1&page_size=5000'


headers = {
    "Authorization": f"Bearer {TOKEN}",
    "User-Agent": "Mozilla/5.0"
}

r = requests.get(url, headers=headers)

# %%
data = r.json()

space = data["results"][0]

count = space.get("current_count", 0)
capacity = space.get("capacity", 1)
percent = (count / capacity) * 100

# %%
# Temperature Pulling
forecast_days = 2 if hour >= 23 else 1
temperatureURL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=37.86866369127376"
    "&longitude=-122.26281535768102"
    "&current=temperature_2m,apparent_temperature,precipitation,relative_humidity_2m"
    "&hourly=temperature_2m,apparent_temperature,precipitation,relative_humidity_2m"
    "&temperature_unit=fahrenheit"
    "&timezone=America/Los_Angeles"
    f"&forecast_days={forecast_days}"
)
temperature = requests.get(temperatureURL).json()

temp = temperature["current"]["temperature_2m"]
feels_like = temperature["current"]["apparent_temperature"]
precipitation = temperature["current"]["precipitation"]
humidity = temperature["current"]["relative_humidity_2m"]

#%%
times = temperature["hourly"]["time"]
 
# Next hour forecast
next_hour = (now + timedelta(hours=1)).strftime("%Y-%m-%dT%H:00")
idx = times.index(next_hour)
temp_forecast        = temperature["hourly"]["temperature_2m"][idx]
feels_like_forecast  = temperature["hourly"]["apparent_temperature"][idx]
precipitation_forecast = temperature["hourly"]["precipitation"][idx]
humidity_forecast    = temperature["hourly"]["relative_humidity_2m"][idx]
 
# 2 hour forecast
next_2hour = (now + timedelta(hours=2)).strftime("%Y-%m-%dT%H:00")
idx2 = times.index(next_2hour)
temp_forecast_2h        = temperature["hourly"]["temperature_2m"][idx2]
feels_like_forecast_2h  = temperature["hourly"]["apparent_temperature"][idx2]
precipitation_forecast_2h = temperature["hourly"]["precipitation"][idx2]
humidity_forecast_2h    = temperature["hourly"]["relative_humidity_2m"][idx2]
 
# 3 hour forecast
next_3hour = (now + timedelta(hours=3)).strftime("%Y-%m-%dT%H:00")
idx3 = times.index(next_3hour)
temp_forecast_3h        = temperature["hourly"]["temperature_2m"][idx3]
feels_like_forecast_3h  = temperature["hourly"]["apparent_temperature"][idx3]
precipitation_forecast_3h = temperature["hourly"]["precipitation"][idx3]
humidity_forecast_3h    = temperature["hourly"]["relative_humidity_2m"][idx3]
# %%
file_path = 'RSF_Dataset.csv'
file_exists = os.path.isfile(file_path)


# %%
row = [
    now.strftime("%Y-%m-%d %H:%M:%S"), # timestamp
    count,                             # current_count
    capacity,                          # capacity
    round(percent, 2),                 # percent_full
    day,                               # weekday (0-6)
    hour,                               # hour (0-23)
    minute,
    temp,
    feels_like,
    precipitation,
    humidity,
    temp_forecast,
    feels_like_forecast,
    precipitation_forecast,
    humidity_forecast,
    temp_forecast_2h,
    feels_like_forecast_2h,
    precipitation_forecast_2h,
    humidity_forecast_2h,
    temp_forecast_3h,
    feels_like_forecast_3h,
    precipitation_forecast_3h,
    humidity_forecast_3h,
]


# %%
with open(file_path, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(['timestamp', 'current_count', 'capacity', 'percent_full', 'weekday', 'hour', 'minute', 'temp', 'feels_like', 'precipitation', 'humidity', 'temperature_forecast', 'feels_like_forecast', 'precipitation_forecast', 'humidity_forecast', 'temperature_forecast_2h', 'feels_like_forecast_2h', 'precipitation_forecast_2h', 'humidity_forecast_2h', 'temperature_forecast_3h', 'feels_like_forecast_3h', 'precipitation_forecast_3h', 'humidity_forecast_3h'])
    
    # Ensure 'row' is defined before this (as you have it)
    writer.writerow(row)

