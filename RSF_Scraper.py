# %%
from datetime import datetime
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

open_now = False

if day <= 4:  # Monday–Friday
    open_now = 7 <= hour < 23

elif day == 5:  # Saturday
    open_now = 8 <= hour < 18

elif day == 6:  # Sunday
    open_now = 8 <= hour < 23

if not open_now:
    print("RSF closed")
    sys.exit(0)



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
    minute
]

# %%
with open(file_path, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(['timestamp', 'current_count', 'capacity', 'percent_full', 'weekday', 'hour', 'minute'])
    
    # Ensure 'row' is defined before this (as you have it)
    writer.writerow(row)

print(f"--- Data successfully saved to: {file_path} ---")

# %%



