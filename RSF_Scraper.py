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
# Pull CURRENT temperature
temperatureURL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=37.86866369127376"
    "&longitude=-122.26281535768102"
    "&current=temperature_2m,apparent_temperature,precipitation,relative_humidity_2m"
    "&temperature_unit=fahrenheit"
    "&timezone=America/Los_Angeles"
    "&forecast_days=1"
)
temperature = requests.get(temperatureURL).json()

temp = temperature["current"]["temperature_2m"]
feels_like = temperature["current"]["apparent_temperature"]
precipitation = temperature["current"]["precipitation"]
humidity = temperature["current"]["relative_humidity_2m"]

#%%
#Get what was forecasted for NOW from (1, 2, 3) hours ago
current_hour_str = now.strftime("%Y-%m-%dT%H:00")
 
def get_past_forecast(offset):
    origin = now - timedelta(hours=offset)
    date_str = origin.strftime("%Y-%m-%d")
    url = (
        "https://historical-forecast-api.open-meteo.com/v1/forecast"
        "?latitude=37.86866369127376"
        "&longitude=-122.26281535768102"
        "&hourly=temperature_2m,apparent_temperature,precipitation,relative_humidity_2m"
        "&temperature_unit=fahrenheit"
        "&timezone=America/Los_Angeles"
        f"&start_date={date_str}&end_date={date_str}"
    )
    data = requests.get(url).json()
    times = data["hourly"]["time"]
    idx = times.index(current_hour_str)
    return {
        "temperature":   data["hourly"]["temperature_2m"][idx],
        "feels_like":    data["hourly"]["apparent_temperature"][idx],
        "precipitation": data["hourly"]["precipitation"][idx],
        "humidity":      data["hourly"]["relative_humidity_2m"][idx],
    }
 
f1 = get_past_forecast(1)
f2 = get_past_forecast(2)
f3 = get_past_forecast(3)
 
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
    f1["temperature"],
    f1["feels_like"],
    f1["precipitation"],
    f1["humidity"],
    f2["temperature"],
    f2["feels_like"],
    f2["precipitation"],
    f2["humidity"],
    f3["temperature"],
    f3["feels_like"],
    f3["precipitation"],
    f3["humidity"],
]


# %%
with open(file_path, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(['timestamp', 'current_count', 'capacity', 'percent_full', 'weekday', 'hour', 'minute', 'temp', 'feels_like', 'precipitation', 'humidity', 'temperature_forecast', 'feels_like_forecast', 'precipitation_forecast', 'humidity_forecast', 'temperature_forecast_2h', 'feels_like_forecast_2h', 'precipitation_forecast_2h', 'humidity_forecast_2h', 'temperature_forecast_3h', 'feels_like_forecast_3h', 'precipitation_forecast_3h', 'humidity_forecast_3h'])
    
    # Ensure 'row' is defined before this (as you have it)
    writer.writerow(row)

