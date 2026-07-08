import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("sho_data.csv");

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.plot(df['t'], df['x'], lw=1, color='blue')
ax1.set_xlabel('t')
ax1.set_ylabel('x')
ax1.set_title("Position")

ax2.plot(df['t'], df['v'], lw=1, color='red')
ax2.set_xlabel('t')
ax2.set_ylabel('v')
ax2.set_title("Velocity")

plt.tight_layout()
plt.show()
