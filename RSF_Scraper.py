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

print(r.status_code)
print(r.text[:500])

# %%
data = r.json()

space = data["results"][0]

count = space.get("current_count", 0)
capacity = space.get("capacity", 1)
percent = (count / capacity) * 100

print("Time:", datetime.now())
print("People:", count)
print("Capacity:", capacity)
print("Percent:", percent)

print(datetime.utcnow().isoformat())

# %%
# Temperature Pulling
temperatureURL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=37.86866369127376"
    "&longitude=-122.26281535768102"
    "&current=temperature_2m,apparent_temperature,precipitation,relative_humidity_2m"
    "&hourly=temperature_2m,apparent_temperature,precipitation,relative_humidity_2m"
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
# Next hour forecast
next_hour = (now + timedelta(hours=1)).strftime("%Y-%m-%dT%H:00")
times = temperature["hourly"]["time"]
idx = times.index(next_hour)

temp_forecast        = temperature["hourly"]["temperature_2m"][idx]
feels_like_forecast  = temperature["hourly"]["apparent_temperature"][idx]
precipitation_forecast = temperature["hourly"]["precipitation"][idx]
humidity_forecast    = temperature["hourly"]["relative_humidity_2m"][idx]
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
]


# %%
with open(file_path, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(['timestamp', 'current_count', 'capacity', 'percent_full', 'weekday', 'hour', 'minute', 'temp', 'feels_like', 'precipitation', 'humidity', 'temperature_forecast', 'feels_like_forecast', 'precipitation_forecast', 'humidity_forecast'])
    
    # Ensure 'row' is defined before this (as you have it)
    writer.writerow(row)

print(f"--- Data saved to: {file_path} ---")

# %%
