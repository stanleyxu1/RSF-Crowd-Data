#%%
import joblib
import pandas as pd
import numpy as np
import datetime
import pytz
from pathlib import Path
 
#%%
script_dir = Path(__file__).parent  
csv_path = script_dir.parent / "RSF_Dataset.csv"
df = pd.read_csv(csv_path)
 
#%%
#Feature Engineering
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["open_hour"] = 7
df["close_hour"] = 23
 
df.loc[df["weekday"].isin([5,6]), "open_hour"] = 8
df.loc[df["weekday"] == 5, "close_hour"] = 18
 
#Account for diff hours in Spring Break
spring_break = {
    "2026-03-23": (8, 20),
    "2026-03-24": (8, 20),
    "2026-03-25": (8, 20),
    "2026-03-26": (8, 20),
    "2026-03-27": (8, 20),
    "2026-03-28": (8, 18),
    "2026-03-29": (8, 20)
}
 
df["date_str"] = df["timestamp"].dt.strftime("%Y-%m-%d")
 
for date, (open_h, close_h) in spring_break.items():
    mask = df["date_str"] == date
    df.loc[mask, "open_hour"] = open_h
    df.loc[mask, "close_hour"] = close_h
 
df["is_open"] = (
    (df["hour"] >= df["open_hour"]) &
    (df["hour"] <= df["close_hour"])
).astype(int)
 
#Lag Features
df["last_percent_full"] = df["percent_full"].shift(1)
df['last_10_mins'] = df['percent_full'].shift(2)
df['last_15_mins'] = df['percent_full'].shift(3)
 
#Time Features
df = df.sort_values("timestamp")
df["hour"] = df["timestamp"].dt.hour
df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["minutes_until_close"] = (df["close_hour"] - df["hour"]) * 60 - df["minute"]
df["weekday_hour"] = df["weekday"] * 24 + df["hour"]
 
#Only Keep Open Times
df = df[df["is_open"] == 1]
 
#%%
#Collect necessary features for models
features = [
    "hour_sin",
    "hour_cos",
    "weekday",
    "week_of_year",
    "minutes_until_close",
    "last_percent_full",
    "weekday_hour",
]
 
featuresXGB = [
    "hour_sin",
    "hour_cos",
    "weekday",
    "week_of_year",
    "minutes_until_close",
    "last_percent_full",
    "weekday_hour",
    "last_10_mins",
    "last_15_mins",
]
 
#%%
#Load Models
xgb_model = joblib.load(script_dir / 'xgb_model.pkl')
model     = joblib.load(script_dir / 'rf_model.pkl')
LinReg    = joblib.load(script_dir / 'linreg_model.pkl')
 
#%%
#Collect future data
tz = pytz.timezone("America/Los_Angeles")
now = datetime.datetime.now(tz)
future_datetime = now + datetime.timedelta(hours=1)
 
future_hour = future_datetime.hour
weekday = future_datetime.weekday()
 
hour_rad = 2 * np.pi * future_hour / 24
hour_sin = np.sin(hour_rad)
hour_cos = np.cos(hour_rad)
 
open_hour = 8 if weekday in [5, 6] else 7
close_hour_future = 18 if weekday == 5 else 23
 
minutes_until_close = (close_hour_future - future_hour) * 60 - future_datetime.minute
week_of_year = future_datetime.isocalendar()[1]
weekday_hour = weekday * 24 + future_hour
last_percent_full = df['percent_full'].iloc[-1]
last_10_mins = df['percent_full'].iloc[-2]
last_15_mins = df['percent_full'].iloc[-3]
 
#Only predict model if gym open
if future_hour < open_hour or future_hour >= close_hour_future:
    print(f"Gym is closed at {future_hour}:00.")
else:
    X_future = pd.DataFrame([[
        hour_sin, hour_cos, weekday, week_of_year,
        minutes_until_close, last_percent_full, weekday_hour
    ]], columns=features)
 
    X_future_XGB = pd.DataFrame([[
        hour_sin, hour_cos, weekday, week_of_year,
        minutes_until_close, last_percent_full, weekday_hour,
        last_10_mins, last_15_mins
    ]], columns=featuresXGB)
 
    #Keep model only at 0% to 100%
    predicted_percent_full     = np.clip(model.predict(X_future), 0, 100)
    LinReg_predicted_percent_full = np.clip(LinReg.predict(X_future), 0, 100)
    XGB_predicted_percent_full = np.clip(xgb_model.predict(X_future_XGB), 0, 100)
 
    #Change README with predictions
    readme_path = script_dir.parent / "README.md"
    marker = "<!-- GYM_PREDICTION -->"
 
    with open(readme_path, "r") as f:
        lines = f.readlines()
 
    future_date_str = future_datetime.strftime("%B %d, %Y")
 
    new_line = (f"{marker}\n"
                f"**Gym Crowdedness Predictor (Next Hour)**\n\n"
                f"XGBoost prediction at {future_hour}:00 on {future_date_str}: {XGB_predicted_percent_full[0]:.1f}%,  \n"
                f"Random Forest prediction at {future_hour}:00 on {future_date_str}: {predicted_percent_full[0]:.1f}%,  \n"
                f"Linear Regression prediction at {future_hour}:00 on {future_date_str}: {LinReg_predicted_percent_full[0]:.1f}%\n")
 
    #Keep graph/images along with new predictions
    for i, line in enumerate(lines):
        if marker in line:
            j = i + 1
            while j < len(lines) and not lines[j].startswith("<!--"):
                j += 1
            image_lines = [l for l in lines[i:j] if l.strip().startswith("![")]
            lines[i:j] = [new_line] + image_lines
            break
 

    with open(readme_path, "w") as f:
        f.writelines(lines)

# %%
#