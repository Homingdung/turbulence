"""
Energy Spectrum Evolution Animation
------------------------------------
Panels: E_u(k), E_B(k), E_A(k), E_total(k), H_cross(k)
Layout: 2×3  (last cell used for time readout)
Background: white, publication-friendly

Usage:
    python turbulence_spectrum_animation.py

Requirements:
    pip install matplotlib numpy pandas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv('spectrum_all.csv')
times  = sorted(df['t'].unique())
k_vals = sorted(df['k'].unique())

def build_array(col):
    arr = np.zeros((len(times), len(k_vals)))
    for i, t in enumerate(times):
        sub = df[df['t'] == t].sort_values('k')
        arr[i] = sub[col].values
    return arr

print("Loading data...")
E_u     = build_array('E_u')
E_B     = build_array('E_B')
E_A     = build_array('E_A')
E_tot   = build_array('E_total_ub')
H_cross = build_array('H_cross')

k = np.array(k_vals)

# ── Figure layout: 2×3 ────────────────────────────────────────────────────────
#   [ E_u ] [ E_B  ] [ E_A  ]
#   [ Etot] [H_cross] [info ]

fig = plt.figure(figsize=(15, 8))
fig.patch.set_facecolor('white')

gs = GridSpec(2, 3, figure=fig,
              left=0.07, right=0.97, top=0.91, bottom=0.09,
              hspace=0.45, wspace=0.32)

ax_u   = fig.add_subplot(gs[0, 0])
ax_B   = fig.add_subplot(gs[0, 1])
ax_A   = fig.add_subplot(gs[0, 2])
ax_tot = fig.add_subplot(gs[1, 0])
ax_hc  = fig.add_subplot(gs[1, 1])
ax_inf = fig.add_subplot(gs[1, 2])   # info / time display panel

PLOT_AXES = [ax_u, ax_B, ax_A, ax_tot, ax_hc]
ALL_AXES  = PLOT_AXES + [ax_inf]

for ax in PLOT_AXES:
    ax.set_facecolor('white')
    ax.tick_params(colors='#333333', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#aaaaaa')
        spine.set_linewidth(0.8)
    ax.grid(True, color='#e0e0e0', linewidth=0.6, linestyle='--', zorder=0)

# info panel — clean, no axes
ax_inf.set_facecolor('white')
ax_inf.set_xticks([]); ax_inf.set_yticks([])
for sp in ax_inf.spines.values():
    sp.set_visible(False)

# ── Colors ─────────────────────────────────────────────────────────────────────
C_u   = '#1f77b4'
C_B   = '#d62728'
C_A   = '#2ca02c'
C_tot = '#7f4f9e'
C_hc  = '#17becf'
C_ref = '#aaaaaa'

# ── k^{-5/3} reference ────────────────────────────────────────────────────────
k_ref    = np.linspace(k[1], k[-1], 200)
k53_norm = E_tot[0, 2] * k[2] ** (5 / 3)
k53      = k53_norm * k_ref ** (-5 / 3)

for ax in [ax_u, ax_B, ax_A, ax_tot]:
    ax.loglog(k_ref, k53, color=C_ref, lw=1.2, ls='--', alpha=0.7,
              label=r'$k^{-5/3}$', zorder=1)

ax_hc.axhline(0, color=C_ref, lw=1.2, ls='--', alpha=0.7, zorder=1)

# ── Animated lines + trails ────────────────────────────────────────────────────
N_TRAIL = 6

def make_trails(ax, color, log=True):
    fn = ax.loglog if log else ax.plot
    return [fn([], [], color=color, lw=0.7, alpha=0)[0] for _ in range(N_TRAIL)]

ln_u,   = ax_u.loglog(  [], [], color=C_u,   lw=2.0, zorder=3, label=r'$E_u(k)$')
ln_B,   = ax_B.loglog(  [], [], color=C_B,   lw=2.0, zorder=3, label=r'$E_B(k)$')
ln_A,   = ax_A.loglog(  [], [], color=C_A,   lw=2.0, zorder=3, label=r'$E_A(k)$')
ln_tot, = ax_tot.loglog([], [], color=C_tot, lw=2.0, zorder=3, label=r'$E_{tot}(k)$')
ln_hc,  = ax_hc.plot(   [], [], color=C_hc,  lw=2.0, zorder=3)

trail_u   = make_trails(ax_u,   C_u)
trail_B   = make_trails(ax_B,   C_B)
trail_A   = make_trails(ax_A,   C_A)
trail_tot = make_trails(ax_tot, C_tot)

# ── Time display inside ax_inf ────────────────────────────────────────────────
time_text = ax_inf.text(0.5, 0.62, '', ha='center', va='center',
                        transform=ax_inf.transAxes,
                        fontsize=22, color='#222222',
                        fontfamily='monospace', fontweight='bold')

# Progress bar drawn as a simple patch inside ax_inf
from matplotlib.patches import FancyArrowPatch, Rectangle
prog_bg_patch  = plt.Rectangle((0.05, 0.32), 0.90, 0.08,
                                transform=ax_inf.transAxes,
                                facecolor='#e8e8e8', edgecolor='#cccccc',
                                linewidth=0.8, zorder=2)
prog_fill = plt.Rectangle((0.05, 0.32), 0.0, 0.08,
                           transform=ax_inf.transAxes,
                           facecolor='#555555', edgecolor='none',
                           zorder=3)
ax_inf.add_patch(prog_bg_patch)
ax_inf.add_patch(prog_fill)

ax_inf.text(0.5, 0.20, 'time progress', ha='center', va='center',
            transform=ax_inf.transAxes,
            fontsize=8, color='#888888')

# ── Labels & titles ────────────────────────────────────────────────────────────
lkw = dict(color='#333333', fontsize=9)
tkw = dict(color='#111111', fontsize=10, fontweight='bold', pad=5)
lgkw = dict(fontsize=8, framealpha=0.85, edgecolor='#cccccc',
            facecolor='white', loc='lower left')

ax_u.set_xlabel('$k$', **lkw);   ax_u.set_ylabel('$E_u(k)$',      **lkw)
ax_B.set_xlabel('$k$', **lkw);   ax_B.set_ylabel('$E_B(k)$',      **lkw)
ax_A.set_xlabel('$k$', **lkw);   ax_A.set_ylabel('$E_A(k)$',      **lkw)
ax_tot.set_xlabel('$k$', **lkw); ax_tot.set_ylabel('$E_{tot}(k)$', **lkw)
ax_hc.set_xlabel('$k$', **lkw);  ax_hc.set_ylabel(r'$H_\times(k)$', **lkw)

ax_u.set_title('Kinetic Energy',          **tkw)
ax_B.set_title('Magnetic Energy',         **tkw)
ax_A.set_title('Vector Potential Energy', **tkw)
ax_tot.set_title('Total Energy',          **tkw)
ax_hc.set_title(r'Cross-helicity $H_\times$', **tkw)

for ax in [ax_u, ax_B, ax_A, ax_tot]:
    ax.legend(**lgkw)

fig.suptitle('MHD Turbulence — Energy Spectrum Evolution',
             fontsize=14, fontweight='bold', color='#111111', y=0.975)

# ── Fixed axis limits ──────────────────────────────────────────────────────────
def ll_lim(arr, pad=3):
    pos = arr[arr > 0]
    return max(pos.min() / pad, 1e-10), arr.max() * pad

for ax, arr in [(ax_u, E_u), (ax_B, E_B), (ax_A, E_A), (ax_tot, E_tot)]:
    ax.set_xlim(k[0] * 0.9, k[-1] * 1.1)
    ax.set_ylim(*ll_lim(arr))

ax_hc.set_xlim(k[0] - 0.5, k[-1] + 0.5)
hc_abs = np.abs(H_cross).max()
ax_hc.set_ylim(-hc_abs * 1.2, hc_abs * 1.2)

ax_inf.set_xlim(0, 1); ax_inf.set_ylim(0, 1)

# ── Update function  (blit=False avoids the NoneType/_get_view bug) ────────────
def update_trails(trails, arr, i):
    for j, tr in enumerate(trails):
        idx = i - (N_TRAIL - j)
        if idx >= 0:
            alpha = (j + 1) / (N_TRAIL + 2) * 0.40
            tr.set_data(k, np.maximum(arr[idx], 1e-10))
            tr.set_alpha(alpha)
        else:
            tr.set_data([], [])

def update(frame):
    i = frame
    ln_u.set_data(  k, np.maximum(E_u[i],   1e-10))
    ln_B.set_data(  k, np.maximum(E_B[i],   1e-10))
    ln_A.set_data(  k, np.maximum(E_A[i],   1e-10))
    ln_tot.set_data(k, np.maximum(E_tot[i], 1e-10))
    ln_hc.set_data( k, H_cross[i])

    update_trails(trail_u,   E_u,   i)
    update_trails(trail_B,   E_B,   i)
    update_trails(trail_A,   E_A,   i)
    update_trails(trail_tot, E_tot, i)

    time_text.set_text(f't = {times[i]:.2f}')

    # update progress bar width in axes-fraction coordinates
    frac = (i + 1) / len(times)
    prog_fill.set_width(0.90 * frac)

# blit=False: avoids AttributeError with extra axes (prog bar patches, fig.text)
ani = animation.FuncAnimation(fig, update,
                               frames=len(times),
                               interval=60,       # ms/frame
                               blit=False)

print("Saving animation — this may take a minute...")
try:
    writer = animation.FFMpegWriter(fps=20, bitrate=2000,
                                    extra_args=['-vcodec', 'libx264'])
    ani.save('spectrum_evolution.mp4', writer=writer, dpi=130,
             savefig_kwargs={'facecolor': 'white'})
    print("✓  Saved  spectrum_evolution.mp4")
except Exception:
    print("ffmpeg not found — saving as GIF instead...")
    ani.save('spectrum_evolution.gif',
             writer='pillow', fps=15, dpi=100,
             savefig_kwargs={'facecolor': 'white'})
    print("✓  Saved  spectrum_evolution.gif")

plt.show()
