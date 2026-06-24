import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ols_cpp_output.csv")

plt.scatter(df['x'], df['y'], label='data', color='red')
plt.plot(df['x'], df['y_hat'], label="C++ OLS", color="green")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()