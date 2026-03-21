readme_path = script_dir.parent / "README.md"
marker = "<!-- GYM_PREDICTION -->"

with open(readme_path, "r") as f:
    lines = f.readlines()

# Prepare new line with both predictions
new_line = (f"{marker} "
            f"Random Forest: {predicted_percent_full[0]:.1f}%, "
            f"Linear Regression: {LinReg_predicted_percent_full[0]:.1f}%\n")

# Replace existing marker line or append
found = False
for i, line in enumerate(lines):
    if marker in line:
        lines[i] = new_line
        found = True
        break

if not found:
    lines.append("\n" + new_line)

with open(readme_path, "w") as f:
    f.writelines(lines)

print("README updated with both model predictions.")

![Gym Crowd Graph](crowd_graph.png)
