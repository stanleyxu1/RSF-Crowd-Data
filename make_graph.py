# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# %%
# Load dataset
df = pd.read_csv("RSF_Dataset.csv")

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Extract hour
df["time_bin"] = df["timestamp"].dt.round("20min")

# Keep only gym open hours
df["hour"] = df["time_bin"].dt.hour
df["minute"] = df["time_bin"].dt.minute

#%%
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
    df.loc[mask, "open_hour"] = None #open_h
    df.loc[mask, "close_hour"] = None #close_h

#Compute is_open
df["is_open"] = (
    (df["hour"] >= df["open_hour"]) &
    (df["hour"] <= df["close_hour"])
).astype(int)

df = df[df['is_open'] == 1]



# %%
# Group data
grouped = df.groupby(["weekday","hour","minute"])["percent_full"].mean().reset_index()


# Sort days and hours
grouped = grouped.sort_values(["weekday","hour","minute"])

# Map weekday names
day_map = {
0: "Monday",
1: "Tuesday",
2: "Wednesday",
3: "Thursday",
4: "Friday",
5: "Saturday",
6: "Sunday"
}


# %%
# Create subplots (7 rows, 1 column)
fig, axes = plt.subplots(7, 1, figsize=(10, 24), sharex=True)

for i, day in enumerate(sorted(grouped["weekday"].unique())):
    data = grouped[grouped["weekday"] == day]

    # convert hour + minute into decimal time
    x = data["hour"] + data["minute"]/60

    axes[i].plot(x, data["percent_full"], marker="o")
    axes[i].set_title(day_map[day])
    axes[i].set_ylabel("% Full")
    ymax = min(105, data["percent_full"].max() + 5)
    axes[i].set_ylim(0, ymax)
    axes[i].set_yticks(np.arange(0, 101, 10))
    axes[i].grid(True)
    axes[i].tick_params(labelbottom=True)


# Common x-axis
axes[-1].set_xlabel("Hour of Day")
axes[-1].set_xticks(range(7,24))

plt.suptitle("RSF Crowd Patterns by Time", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])

# Save graph
plt.savefig("crowd_graph.png", bbox_inches="tight")
# %%
