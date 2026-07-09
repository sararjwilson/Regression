import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ridge_cpp_output.csv")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df["x"], df["y"], s=20, alpha=0.6, label="observed y (noisy)", color="magenta")
ax.plot(df["x"], df["yhat"], color="orange", linewidth=2, label="ridge fit ")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Ridge Regression Fit")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("ridge_plot.png", dpi=150)
plt.show()