# %%
import pandas as pd
import matplotlib.pyplot as plt


# %%
# Load dataset
df = pd.read_csv("RSF_Dataset.csv")

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Extract hour
df["hour"] = df["timestamp"].dt.hour

# Keep only gym open hours
df = df[(df["hour"] >= 7) & (df["hour"] <= 23)]


# %%
# Group data
grouped = df.groupby(["weekday","hour"])["percent_full"].mean().reset_index()

# Sort days and hours
grouped = grouped.sort_values(["weekday","hour"])

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
# Plot
plt.figure(figsize=(10,6))

for day in sorted(grouped["weekday"].unique()):
    data = grouped[grouped["weekday"] == day]
    plt.plot(data["hour"], data["percent_full"], marker="o", label=day_map[day])

plt.xlabel("Hour of Day")
plt.ylabel("Average % Full")
plt.title("RSF Crowd Patterns by Hour")

plt.xticks(range(7,24))
plt.grid(True)
plt.legend()

# Save graph (important for GitHub Actions)
plt.savefig("crowd_graph.png")

# %%



