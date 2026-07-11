import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("wsindy_heat_data.csv");

grid = df.pivot(index='x', columns='t', values='u')
x = grid.index.values
t = grid.columns.values
u = grid.values

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(10,7))

im = ax1.pcolormesh(t, x, u, shading='auto', cmap='plasma')
ax1.set_xlabel('t')
ax1.set_ylabel('x')
ax1.set_title('u(x,t)')
fig.colorbar(im, ax=ax1, label='u')

n = 10;
snapshot_indices = np.linspace(0, len(t) - 1, n, dtype=int)

for i in snapshot_indices:
    ax2.plot(x, u[:, i], lw=1.2, label=f"t={t[i]:.5f}")

ax2.set_xlabel('x')
ax2.set_ylabel('u')
ax2.set_title('Snapshots Over Time')
ax2.legend(fontsize=8)

plt.tight_layout()
plt.show()

