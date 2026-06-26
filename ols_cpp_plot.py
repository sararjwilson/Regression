import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ols_cpp_output.csv")
df = df.sort_values("x")

plt.scatter(df['x'], df['y'], label='data', color='red')
plt.plot(df['x'], df['yhat'], label="C++ OLS", color="green")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()