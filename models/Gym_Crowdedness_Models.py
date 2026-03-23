# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# %%
script_dir = Path(__file__).parent  # models/
csv_path = script_dir.parent / "RSF_Dataset.csv" 
df = pd.read_csv(csv_path)

# %%
df

# %%
df["open_hour"] = 7
df["close_hour"] = 23

# weekend opening time
df.loc[df["weekday"].isin([5,6]), "open_hour"] = 8

# Saturday closing time
df.loc[df["weekday"] == 5, "close_hour"] = 18

df["is_open"] = (
    (df["hour"] >= df["open_hour"]) &
    (df["hour"] < df["close_hour"])
).astype(int)

df["last_percent_full"] = df["percent_full"].shift(1)



# %%
#Data cleaning with time-series values
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")
df["hour"] = df["timestamp"].dt.hour
df["month"] = df["timestamp"].dt.month
df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)

df["is_weekend"] = df["weekday"].isin([5,6]).astype(int)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df = df[df["is_open"] == 1]

# %%
df["minutes_until_close"] = (
    (df["close_hour"] - df["hour"]) * 60
    - df["minute"]
)

df["weekday_hour"] = df["weekday"] * 24 + df["hour"]

# %%
corr_target = df.corr(numeric_only=True)["percent_full"].sort_values(ascending=False)

print(corr_target)


# %%
features = [
    "hour_sin",
    "hour_cos",
    "weekday",
    "week_of_year",
    "minutes_until_close",
    "last_percent_full",
    "weekday_hour"
]


# %%
X = df.iloc[1:][features]
y = df[1:]["percent_full"]

# %%
X_train, X_test, y_train, y_test = train_test_split(
X, y, train_size=0.8, random_state=42)


# %%
#Naive baseline, all predictions are overall average
mean_value = y_train.mean()
baseline_preds = [mean_value] * len(y_test)

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, baseline_preds))

# %%
LinReg = LinearRegression()
LinReg.fit(X_train, y_train)

# %%
predictions = LinReg.predict(X_test)
mae_test = mean_absolute_error(y_test, predictions)
rmse_test = np.sqrt(mean_squared_error(y_test, predictions))


# %%
print("Mean Absolute Error (Testing Data):", mae_test)
print("RMSE (Testing Data):", rmse_test)


# %%
predictions_training = LinReg.predict(X_train)
mae_train = mean_absolute_error(y_train, predictions_training)
rmse_train = np.sqrt(mean_squared_error(y_train, predictions_training))
print("Mean Absolute Error (Training Data):", mae_train)
print("RMSE (Training Data):", rmse_train)



# %%
plt.scatter(predictions, y_test)
plt.xlabel("Predicted % Full")
plt.ylabel("Actual % Full")
plt.title("Predicted vs Actual Gym Occupancy")
plt.show()

# %%
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("Mean Absolute Error:", mae)
print("RMSE:", rmse)

# %%
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "feature": features,
    "importance": importance
}).sort_values(by="importance", ascending=False)

print(feature_importance)
import matplotlib.pyplot as plt

feature_importance.plot(
    x="feature",
    y="importance",
    kind="bar",
    legend=False
)

plt.title("Feature Importance")
plt.ylabel("Importance")
plt.show()

# %%
import datetime, pytz

# Use your local timezone instead of UTC
tz = pytz.timezone("America/Los_Angeles")
now = datetime.datetime.now(tz)
future_datetime = now + datetime.timedelta(hours=1)

future_hour = future_datetime.hour  # safer than now.hour + 1 (avoids 24)
weekday = future_datetime.weekday() # derive from actual current time, not df

# Hour encoding (cyclical)
hour_rad = 2 * np.pi * future_hour / 24
hour_sin = np.sin(hour_rad)
hour_cos = np.cos(hour_rad)

# Minutes until close
open_hour = 8 if weekday in [5, 6] else 7
close_hour_future = 18 if weekday == 5 else 23

minutes_until_close = (close_hour_future - future_hour) * 60 - future_datetime.minute
week_of_year = future_datetime.isocalendar()[1]
weekday_hour = weekday * 24 + future_hour
last_percent_full = df['percent_full'].iloc[-1]
print(f"Local time: {now}, future_hour: {future_hour}, weekday: {weekday}")
if future_hour < open_hour or future_hour >= close_hour_future:
    print(f"Gym is closed at {future_hour}:00. No prediction made.")
else:
    # Everything from X_future creation onwards
    X_future = pd.DataFrame([[
        hour_sin,
        hour_cos,
        weekday,
        week_of_year,
        minutes_until_close,
        last_percent_full,
        weekday_hour
    ]], columns=features)

    predicted_percent_full = model.predict(X_future)
    LinReg_predicted_percent_full = LinReg.predict(X_future)

    readme_path = script_dir.parent / "README.md"
    marker = "<!-- GYM_PREDICTION -->"

    with open(readme_path, "r") as f:
        lines = f.readlines()

    future_date_str = future_datetime.strftime("%B %d, %Y")

    new_line = (f"{marker}\n"
            f"**Gym Crowdedness Predictor (Next Hour)**\n\n"
            f"Random Forest prediction at {future_hour}:00 on {future_date_str}: {predicted_percent_full[0]:.1f}%,  \n"
            f"Linear Regression prediction at {future_hour}:00 on {future_date_str}: {LinReg_predicted_percent_full[0]:.1f}%\n")

    found = False
    for i, line in enumerate(lines):
        if marker in line:
            j = i + 1
            while j < len(lines) and not lines[j].startswith("<!--"):
                j += 1
            image_lines = [l for l in lines[i:j] if l.strip().startswith("![")]
            lines[i:j] = [new_line] + image_lines
            found = True
            break

    if not found:
        lines.append("\n" + new_line)

    with open(readme_path, "w") as f:
        f.writelines(lines)
# %%
