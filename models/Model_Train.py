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
##Change timestamp to datetime type & establish open and closing hours
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
    df.loc[mask, "open_hour"] = None
    df.loc[mask, "close_hour"] = None

#Compute is_open
df["is_open"] = (
    (df["hour"] >= df["open_hour"]) &
    (df["hour"] <= df["close_hour"])
).astype(int)

#%%
#Only consider OPEN times
df = df[df["is_open"] == 1]
#%%
#Lag Features, percent full
df["last_5_mins"] = df["percent_full"].shift(1)
df['last_10_mins'] = df['percent_full'].shift(2)
df['last_15_mins'] = df['percent_full'].shift(3)

#%%
#Rolling mean and std
df["rolling_mean_15"] = (
    df["percent_full"]
    .rolling(window=3, min_periods=1)
    .mean()
)

    # Rolling std (volatility in last 15 mins)
df["rolling_std_15"] = (
    df["percent_full"]
    .rolling(window=3, min_periods=2)
    .std()
)

    # Optional: slightly longer context (30 mins)
df["rolling_mean_30"] = (
    df["percent_full"]
    .rolling(window=6, min_periods=1)
    .mean()
)

df["rolling_std_30"] = (
    df["percent_full"]
    .rolling(window=6, min_periods=2)
    .std()
)

# %%
#Turning Dates to Timeseries type
df = df.sort_values("timestamp")
df["hour"] = df["timestamp"].dt.hour
df["month"] = df["timestamp"].dt.month
df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)

#create is_weekend col
df["is_weekend"] = df["weekday"].isin([5,6]).astype(int)

#Use sin and cos for hours (time circular)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)


# %%
#create minutes_until_close col
df["minutes_until_close"] = (
    (df["close_hour"] - df["hour"]) * 60
    - df["minute"]
)

#What hour in the entire week
df["weekday_hour"] = df["weekday"] * 24 + df["hour"]

#%%
#trend
df['delta_5'] = df['last_percent_full'] - df['last_10_mins']
df['delta_10'] = df['last_10_mins'] - df['last_15_mins']

# %%
#See feature correlation
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
# time
    "hour_sin",
    "hour_cos",
    "weekday",
    "weekday_hour",
    "minutes_until_close",

    # recency
    "last_percent_full",
    "last_10_mins",
    "last_15_mins",

    #trend
    "delta_5",
    "delta_10",

    #Rolling data
    "rolling_mean_15",
    "rolling_mean_30",
    "rolling_std_15",
    "rolling_std_30",
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
 

#%%
results = {}
models = {}

#3 Different XGB models for diff time horizons
prediction_horizons = {
    15: "15 minutes ahead",
    30: "30 minutes ahead",
    45: "45 minutes ahead"
}
# %%
# XGBoost
for horizon, description in prediction_horizons.items():
    print(f"Training model for {description}")
    
    # Create a copy for this horizon
    df_horizon = df.copy()
    
    # At time t, we want to predict percent_full at time t+horizon
    df_horizon[f"target_{horizon}"] = df_horizon["percent_full"].shift(-horizon)
    
    # Drop rows where target is NaN (last 'horizon' rows have no future value)
    # This automatically handles the issue where we can't predict beyond the dataset
    df_horizon = df_horizon.dropna(subset=[f"target_{horizon}"])
    
    #Only use features that don't have NaN values
    df_horizon = df_horizon.dropna(subset=featuresXGB)
 
    # Prepare data
    X = df_horizon[featuresXGB]
    y = df_horizon[f"target_{horizon}"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Baseline (mean prediction)
    mean_value = y_train.mean()
    baseline_preds = np.full(len(y_test), mean_value)
    baseline_mae = mean_absolute_error(y_test, baseline_preds)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
    print(f"\nBaseline MAE:  {baseline_mae:.4f}")
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    
    # Train XGBoost model
    xgb_model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    xgb_model.fit(X_train, y_train)
    
    # Predictions
    train_preds = xgb_model.predict(X_train)
    test_preds = xgb_model.predict(X_test)
    
    # Evaluation metrics
    mae_train = mean_absolute_error(y_train, train_preds)
    rmse_train = np.sqrt(mean_squared_error(y_train, train_preds))
    
    mae_test = mean_absolute_error(y_test, test_preds)
    rmse_test = np.sqrt(mean_squared_error(y_test, test_preds))
    
    print(f"\nTraining Results:")
    print(f"  MAE:  {mae_train:.4f}")
    print(f"  RMSE: {rmse_train:.4f}")
    
    print(f"\nTest Results:")
    print(f"  MAE:  {mae_test:.4f}")
    print(f"  RMSE: {rmse_test:.4f}")
    
    # Store results
    results[horizon] = {
        "mae_train": mae_train,
        "rmse_train": rmse_train,
        "mae_test": mae_test,
        "rmse_test": rmse_test,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "improvement": ((baseline_mae - mae_test) / baseline_mae * 100)
    }
    
    models[horizon] = xgb_model
    
    # Feature importance
    importance_df = pd.DataFrame({
        "feature": featuresXGB,
        "importance": xgb_model.feature_importances_
    }).sort_values(by="importance", ascending=False)
    
    print(f"\nTop 8 Features ({horizon}min ahead):")
    print(importance_df.head(8).to_string(index=False))
    
    
    # Visualization 2: Feature Importance
    plt.figure(figsize=(10, 7))
    importance_df.head(12).sort_values('importance').plot(
        x="feature",
        y="importance",
        kind="barh",
        legend=False
    )
    plt.title(f"Feature Importance ({horizon}min ahead)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()
 
# %%
# SUMMARY COMPARISON
print(f"\n\n{'='*80}")
print("SUMMARY: Model Performance Across All Horizons")
print(f"{'='*80}")
 
summary_df = pd.DataFrame(results).T
print(summary_df[["mae_train", "mae_test", "rmse_test", "improvement"]].round(4))
 
# %%
# Visualization 3: Model comparison across horizons
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
 
horizons = list(results.keys())
horizons_str = [f"{h}m" for h in horizons]
 
# MAE comparison
ax = axes[0, 0]
baselines = [results[h]['baseline_mae'] for h in horizons]
test_maes = [results[h]['mae_test'] for h in horizons]
x = np.arange(len(horizons))
width = 0.35
ax.bar(x - width/2, baselines, width, label='Baseline (Mean)', alpha=0.8)
ax.bar(x + width/2, test_maes, width, label='XGBoost', alpha=0.8)
ax.set_xlabel("Prediction Horizon")
ax.set_ylabel("MAE")
ax.set_title("Mean Absolute Error Comparison")
ax.set_xticks(x)
ax.set_xticklabels(horizons_str)
ax.legend()
ax.grid(True, alpha=0.3)
 
# RMSE comparison
ax = axes[0, 1]
test_rmses = [results[h]['rmse_test'] for h in horizons]
ax.plot(horizons_str, test_rmses, marker='o', linewidth=2, markersize=8, color='steelblue')
ax.set_xlabel("Prediction Horizon")
ax.set_ylabel("RMSE")
ax.set_title("Root Mean Square Error")
ax.grid(True, alpha=0.3)
 
 
# Improvement over baseline
ax = axes[1, 1]
improvements = [results[h]['improvement'] for h in horizons]
colors = ['green' if imp > 0 else 'red' for imp in improvements]
ax.bar(horizons_str, improvements, color=colors, alpha=0.8)
ax.set_xlabel("Prediction Horizon")
ax.set_ylabel("Improvement (%)")
ax.set_title("% Improvement Over Baseline")
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3)
 
plt.tight_layout()
plt.show()
 
# %%
# Save models
print("Saving Models")
 
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
 
for horizon, model in models.items():
    save_path = f"xgb_model_{horizon}min.pkl"
    joblib.dump(model, save_path)
 
print("\nAll models trained and saved successfully")
print(f"Models available for {list(models.keys())} minute predictions")
# %%
