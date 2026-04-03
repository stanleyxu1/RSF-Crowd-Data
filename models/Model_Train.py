# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
import joblib

# %%
script_dir = Path(__file__).parent  # models/
csv_path = script_dir.parent / "RSF_Dataset.csv" 
df = pd.read_csv(csv_path)

# %%
df["timestamp"] = pd.to_datetime(df["timestamp"])
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
df["last_percent_full"] = df["percent_full"].shift(1)
df['last_10_mins'] = df['percent_full'].shift(2)
df['last_15_mins'] = df['percent_full'].shift(3)

# %%
#Turning Dates to Timeseries type
df = df.sort_values("timestamp")
df["hour"] = df["timestamp"].dt.hour
df["month"] = df["timestamp"].dt.month
df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)

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

# %%
X = df.iloc[1:][features]
y = df[1:]["percent_full"]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
#Naive baseline, all predictions are simply overall average
mean_value = y_train.mean()
baseline_preds = [mean_value] * len(y_test)
print("Baseline MAE:", mean_absolute_error(y_test, baseline_preds))
 
# %%
#Fit Linear Regression Model
LinReg = LinearRegression()
LinReg.fit(X_train, y_train)

# %%
#Predict TEST data with LinReg Model and get Error Metrics
predictions = LinReg.predict(X_test)
mae_test_lr = mean_absolute_error(y_test, predictions)
mse_test_lr = np.sqrt(mean_squared_error(y_test, predictions))
print("Mean Absolute Error (Testing Data):", mae_test_lr)
print("RMSE (Testing Data):", mse_test_lr)

# %%
#Predict TRAIN data with LinReg Model and get Error Metrics
predictions_training = LinReg.predict(X_train)
mae_train = mean_absolute_error(y_train, predictions_training)
rmse_train = np.sqrt(mean_squared_error(y_train, predictions_training))
print("Mean Absolute Error (Training Data):", mae_train)
print("RMSE (Training Data):", rmse_train)

# %%
#Predicted v Actual Graph on Linear Regression
plt.scatter(predictions, y_test)
plt.xlabel("Predicted % Full")
plt.ylabel("Actual % Full")
plt.title("Predicted vs Actual Gym Occupancy")
plt.title("Predicted v Actual (Linear Regression)")
plt.show()
 
# %%
#Fit Random Forest Regressor Model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)
#%%
#Predict TEST data with Random Forest Model and get Error Metrics
predictions = model.predict(X_test)

# evaluation
mae_test_rf = mean_absolute_error(y_test, predictions)
rmse_test_rf = np.sqrt(mean_squared_error(y_test, predictions))

print("Mean Absolute Error (Random Forest Testing Data):", mae_test_rf)
print("RMSE (Random Forest Testing Data):", rmse_test_rf)


# %%
#Find which features most important to model
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "feature": features,
    "importance": importance
}).sort_values(by="importance", ascending=False)

print(feature_importance)

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
# XGBoost
X_XGB = df.iloc[1:][featuresXGB]
y_XGB = df[1:]["percent_full"]

X_train_XGB, X_test_XGB, y_train_XGB, y_test_XGB = train_test_split(X_XGB, y_XGB, test_size=0.2, random_state=42)

#%%
xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train_XGB, y_train_XGB)
xgb_predictions = xgb_model.predict(X_test_XGB)

# Evaluation
mae_test_xgb = mean_absolute_error(y_test_XGB, xgb_predictions)
mse_test_xgb = np.sqrt(mean_squared_error(y_test_XGB, xgb_predictions))

print("Mean Absolute Error (XGBoost):", mae_test_xgb)
print("RMSE (XGBoost):", mse_test_xgb)


importance_XGB = xgb_model.feature_importances_

feature_importance_XGB = pd.DataFrame({
    "feature": featuresXGB,
    "importance": importance_XGB
}).sort_values(by="importance", ascending=False)

print(feature_importance_XGB)

feature_importance_XGB.plot(
    x="feature",
    y="importance",
    kind="bar",
    legend=False
)
plt.title("Feature Importance (XGBoost)")
plt.ylabel("Importance")
plt.show()
 
# %%
# Save models
joblib.dump(xgb_model, 'models/xgb_model.pkl')
joblib.dump(model, 'models/rf_model.pkl')
joblib.dump(LinReg, 'models/linreg_model.pkl')
 
