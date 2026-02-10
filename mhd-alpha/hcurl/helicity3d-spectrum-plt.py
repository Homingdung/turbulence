# plot helicity spectrum.py
import pandas as pd
import matplotlib.pyplot as plt
import os

# ====== 1. 路径 ======
data_path = "output/spectrum_all.csv"
mesh_info_path = "output/mesh_info.csv"
out_dir = "output"
os.makedirs(out_dir, exist_ok=True)

# ====== 2. 读 spectrum 数据 ======
df = pd.read_csv(data_path)

if "E_total" not in df.columns:
    df["E_total"] = df["E_u"] + df["E_B"]

# ====== 3. 选最后一个时间步 ======
t = df["t"].max()
#t_plot = 1.5
#t_vals = df["t"].unique()
#t = t_vals[(abs(t_vals - t_plot)).argmin()]
df_t = df[df["t"] == t]

k = df_t["k"].values
E_u = df_t["E_u"].values
E_B = df_t["E_B"].values
E_tot = df_t["E_total"].values
H_mag = df_t["H_mag"].values
H_cross = df_t["H_cross"].values

# ====== 4. 读 k_alpha ======
mesh_info = pd.read_csv(mesh_info_path)
k_alpha = mesh_info["k_alpha"].iloc[0]

# ====== 5. 画 spectrum ======
plt.figure(figsize=(6,4))
plt.plot(k, H_mag, "-", label="MagneticHelicity")
plt.plot(k, H_cross, "-", label="CrossHelicity")

# ---- k^{-5/3} 参考线（对齐中间波数）----
k0 = k[len(k)//2]
C = E_tot[len(k)//2] * (k0**(5/3))
#plt.loglog(k, C * k**(-5/3), ":", linewidth=2,label=r"$k^{-5/3}$")

# ---- k_alpha 竖线 ----
plt.axvline(
    k_alpha,
    color="red",
    linestyle="--",
    linewidth=2,
    label=rf"$k_\alpha = {k_alpha:.2f}$"
)

plt.xlabel(r"$k$")
plt.ylabel(r"$E(k)$")
plt.title(f"Spectrum at t = {t:.3f}")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()

# ====== 6. 保存 + 显示 ======
fig_path = os.path.join(out_dir, f"spectrum_t={t:.3f}.png")
plt.savefig(fig_path, dpi=300)

plt.show()
plt.close()

print(f"Saved spectrum plot to: {fig_path}")

