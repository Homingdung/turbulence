import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("output/data.csv")

plt.plot(df["t"], df["helicity"], label="Helicity")
plt.plot(df["t"], df["energy"], label="Energy")

plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Helicity and Energy vs Time")
plt.legend()
#plt.grid(True)

plt.savefig("helicity_energy.png", dpi=300)
plt.show()

