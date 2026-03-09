import numpy as np
import pandas as pd
from tabulate import tabulate

df = pd.read_csv("all_errors.csv")
df = df.sort_values("dt").reset_index(drop=True)

rates_u, rates_P, rates_B = ["-"], ["-"], ["-"]
for i in range(1, len(df)):
    r = lambda col: np.log(df[col][i] / df[col][i-1]) / np.log(df["dt"][i] / df["dt"][i-1])
    rates_u.append(f"{r('error_u'):.2f}")
    rates_P.append(f"{r('error_p'):.2f}")
    rates_B.append(f"{r('error_B'):.2f}")

table_data = []
for i in range(len(df)):
    table_data.append([
        f"{df['dt'][i]:.4e}",
        f"{df['error_u'][i]:.4e}", rates_u[i],
        f"{df['error_p'][i]:.4e}", rates_P[i],
        f"{df['error_B'][i]:.4e}", rates_B[i],
    ])

headers = ["dt", "Error (u)", "Rate (u)", "Error (P)", "Rate (P)", "Error (B)", "Rate (B)"]

print("\nTemporal Convergence Results:")
print(tabulate(table_data, headers=headers))
print()
print(tabulate(table_data, headers=headers, tablefmt="latex"))
