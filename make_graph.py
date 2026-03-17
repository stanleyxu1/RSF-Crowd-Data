# %%
import pandas as pd
import matplotlib.pyplot as plt


# %%
# Load dataset
df = pd.read_csv("RSF_Dataset.csv")

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Extract hour
df["time_bin"] = df["timestamp"].dt.round("30min")

# Keep only gym open hours
df["hour"] = df["time_bin"].dt.hour
df["minute"] = df["time_bin"].dt.minute
df = df[(df["hour"] >= 7) & (df["hour"] <= 23)]


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
fig, axes = plt.subplots(7, 1, figsize=(10,18), sharex=True)

for i, day in enumerate(sorted(grouped["weekday"].unique())):
    data = grouped[grouped["weekday"] == day]

    # convert hour + minute into decimal time
    x = data["hour"] + data["minute"]/60

    axes[i].plot(x, data["percent_full"], marker="o")
    axes[i].set_title(day_map[day])
    axes[i].set_ylabel("% Full")
    axes[i].set_ylim(data["percent_full"].min()-5, data["percent_full"].max()+10)
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
