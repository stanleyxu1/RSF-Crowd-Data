# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor
import joblib
import datetime
import pytz

# %%
script_dir = Path(__file__).parent  # models/
csv_path = script_dir.parent / "RSF_Dataset.csv" 
df = pd.read_csv(csv_path)

# %%
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["open_hour"] = 7
df["close_hour"] = 23

# weekend opening time
df.loc[df["weekday"].isin([5,6]), "open_hour"] = 8

# Saturday closing time
df.loc[df["weekday"] == 5, "close_hour"] = 18

#SPRING BREAK HOURS
spring_break = {
    "2026-03-23": (8, 20),
    "2026-03-24": (8, 20),
    "2026-03-25": (8, 20),
    "2026-03-26": (8, 20),
    "2026-03-27": (8, 20),
    "2026-03-28": (8, 18),
    "2026-03-29": (8, 20)
}

# Apply spring break overrides
df["date_str"] = df["timestamp"].dt.strftime("%Y-%m-%d")

for date, (open_h, close_h) in spring_break.items():
    mask = df["date_str"] == date
    df.loc[mask, "open_hour"] = open_h
    df.loc[mask, "close_hour"] = close_h

#Compute is_open
df["is_open"] = (
    (df["hour"] >= df["open_hour"]) &
    (df["hour"] <= df["close_hour"])
).astype(int)

#%%
#Lag Features, percent full
df["last_5_mins"] = df.groupby(df["timestamp"].dt.date)["percent_full"].shift(1)
df["last_10_mins"] = df.groupby(df["timestamp"].dt.date)["percent_full"].shift(2)
df["last_15_mins"] = df.groupby(df["timestamp"].dt.date)["percent_full"].shift(3)

# %%
#Rolling mean and std
df["rolling_mean_15"] = (
    df["percent_full"]
    .shift(1)
    .rolling(window=3, min_periods=1)
    .mean()
)

    # Rolling std (volatility in last 15 mins)
df["rolling_std_15"] = (
    df["percent_full"]
    .shift(1)
    .rolling(window=3, min_periods=2)
    .std()
)

    # Optional: slightly longer context (30 mins)
df["rolling_mean_30"] = (
    df["percent_full"]
    .shift(1)
    .rolling(window=6, min_periods=1)
    .mean()
)

df["rolling_std_30"] = (
    df["percent_full"]
    .shift(1)
    .rolling(window=6, min_periods=2)
    .std()
)

# %%
#Turning Dates to Timeseries type
df = df.sort_values("timestamp")
df["month"] = df["timestamp"].dt.month
df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)
df["minute"] = df["timestamp"].dt.minute

#Is weekend
df["is_weekend"] = df["weekday"].isin([5,6]).astype(int)

#Use sin and cos for hours (time is circular)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

#Only take open values
df = df[df["is_open"] == 1]

# %%
df["minutes_until_close"] = (
    (df["close_hour"] - df["hour"]) * 60
    - df["minute"]
)

#What hour in the entire week
df["weekday_hour"] = df["weekday"] * 24 + df["hour"]

#%%
# Trend features
df['delta_5'] = df['last_5_mins'] - df['last_10_mins']
df['delta_10'] = df['last_10_mins'] - df['last_15_mins']
# %%
corr_target = df.corr(numeric_only=True)["percent_full"].sort_values(ascending=False)

print(corr_target)
 
#%%
features = [
    "hour_sin",
    "hour_cos",
    "weekday",
    "week_of_year",
    "minutes_until_close",
    "last_5_mins",
    "weekday_hour",
]
 
featuresXGB = [
    # time
    "hour_sin",
    "hour_cos",
    "weekday",
    "weekday_hour",              # ← week_of_year removed
    "minutes_until_close",
 
    # recency
    "last_5_mins",
    "last_10_mins",
    "last_15_mins",
 
    # trend
    "delta_5",
    "delta_10",
 
    # Rolling data
    "rolling_mean_15",
    "rolling_mean_30",
    "rolling_std_15",
    "rolling_std_30",
]
 
#%%
#Load Models
xgb_model_15 = joblib.load(script_dir / 'xgb_model_15min.pkl')
xgb_model_30 = joblib.load(script_dir / 'xgb_model_30min.pkl')
xgb_model_45 = joblib.load(script_dir / 'xgb_model_45min.pkl')


# Use your local timezone instead of UTC
tz = pytz.timezone("America/Los_Angeles")
now = datetime.datetime.now(tz)

# Prepare predictions for multiple horizons
predictions_by_horizon = {}
models_by_horizon = {15: xgb_model_15, 30: xgb_model_30, 45: xgb_model_45}


for horizon_mins, xgb_model in models_by_horizon.items():

    future_datetime = now + datetime.timedelta(minutes=horizon_mins)
    future_hour = future_datetime.hour
    weekday = future_datetime.weekday()
    
    # Check if gym will be open at that time
    open_hour = 8 if weekday in [5, 6] else 7
    close_hour = 18 if weekday == 5 else 23
    
    # Check spring break hours
    future_date_str_check = future_datetime.strftime("%Y-%m-%d")
    if future_date_str_check in spring_break:
        open_hour, close_hour = spring_break[future_date_str_check]
    
    # Hour encoding (cyclical)
    hour_rad = 2 * np.pi * future_hour / 24
    hour_sin = np.sin(hour_rad)
    hour_cos = np.cos(hour_rad)
    
    # Minutes until close
    minutes_until_close = (close_hour - future_hour) * 60 - future_datetime.minute
    week_of_year = future_datetime.isocalendar()[1]
    weekday_hour = weekday * 24 + future_hour
    
    # Get most recent lag features from dataframe
    last_5_mins = df['percent_full'].iloc[-1]
    last_10_mins = df['percent_full'].iloc[-2] if len(df) > 1 else last_5_mins
    last_15_mins = df['percent_full'].iloc[-3] if len(df) > 2 else last_5_mins
    
    # Trend features
    delta_5 = last_5_mins - last_10_mins
    delta_10 = last_10_mins - last_15_mins
    
    # Rolling stats
    recent15 = df['percent_full'].iloc[-4:-1]
    rolling_mean_15 = recent15.mean()
    rolling_std_15 = recent15.std()

    recent30 = df['percent_full'].iloc[-7:-1]
    rolling_mean_30 = recent30.mean()
    rolling_std_30 = recent30.std()
    
    # Check if gym is open
    if future_hour < open_hour or future_hour >= close_hour:
        predictions_by_horizon[horizon_mins] = {
            'is_open': False,
            'time': future_datetime,
            'prediction': None
        }
    else:
        # Create feature dataframe
        X_future_XGB = pd.DataFrame([[
            hour_sin,
            hour_cos,
            weekday,
            weekday_hour,
            minutes_until_close,
            last_5_mins,
            last_10_mins,
            last_15_mins,
            delta_5,
            delta_10,
            rolling_mean_15,
            rolling_mean_30,
            rolling_std_15,
            rolling_std_30
        ]], columns=featuresXGB)
        
        # Predict and clip to 0-100
        prediction = np.clip(xgb_model.predict(X_future_XGB)[0], 0, 100)

        predictions_by_horizon[horizon_mins] = {
            'is_open': True,
            'time': future_datetime,
            'prediction': prediction
        }

    #README update
readme_path = script_dir.parent / "README.md"
marker = "<!-- GYM_PREDICTION -->"
 
if readme_path.exists():
    with open(readme_path, "r") as f:
        lines = f.readlines()
    
    # Build prediction text with all horizons
    prediction_lines = [f"{marker}"]
    prediction_lines.append("**Gym Crowdedness Predictor**")
    prediction_lines.append("")
    
    for horizon in [15, 30, 45]:
        if horizon in predictions_by_horizon:
            pred_data = predictions_by_horizon[horizon]
            time_str = pred_data['time'].strftime("%H:%M")
            
            if pred_data['is_open']:
                prediction_lines.append(f"**{horizon}min ahead** ({time_str}): {pred_data['prediction']:.1f}%")
            else:
                prediction_lines.append(f"**{horizon}min ahead** ({time_str}): Gym closed")
    
            if horizon != 45:
                prediction_lines.append("")
    prediction_lines.append("")
    prediction_lines.append(f"*Last updated: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}*")
    prediction_lines.append("")  
    
    # Join with newlines
    prediction_text = "\n".join(prediction_lines) + "\n"
    
    print("Preview of README update:")
    print(prediction_text)

    # Find and replace or append
    found = False
    for i, line in enumerate(lines):
        if marker in line:
            j = i + 1
            while j < len(lines) and not lines[j].startswith("<!--"):
                j += 1
            # Preserve any images after the marker
            image_lines = [l for l in lines[i:j] if l.strip().startswith("![")]
            lines[i:j] = [prediction_text] + image_lines
            found = True
            break
    
    if not found:
        lines.append("\n" + prediction_text)
    
    with open(readme_path, "w") as f:
        f.writelines(lines)
# %%
print("CSV exists:", csv_path.exists())
print("DF length:", len(df))
print("Last percent_full values:", df['percent_full'].tail())
# %%
