# src/analysis.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

ARENA_CM = 150  # keep in sync with io_axona.py

def plot_arena(pos_df):
    plt.figure(figsize=(6,6))
    k = pos_df[pos_df["keep"]]
    plt.plot(k["x"], k["y"], lw=0.5)
    plt.plot([0,ARENA_CM,ARENA_CM,0,0],[0,0,ARENA_CM,ARENA_CM,0], lw=1.2)
    ax = plt.gca(); ax.set_aspect("equal")
    ax.set(xlabel="x (cm)", ylabel="y (cm)", xlim=(0,ARENA_CM), ylim=(0,ARENA_CM))
    plt.title("Arena & trajectory (speed-filtered)")
    plt.show()

def occupancy_map(pos_df, bins=40):
    k = pos_df[pos_df["keep"]]
    H, xe, ye = np.histogram2d(k["x"], k["y"], bins=bins, range=[[0,ARENA_CM],[0,ARENA_CM]])
    # 50Hz tracker → seconds
    occ = (H / 50.0).T
    return occ, xe, ye

def center_mask(bins, center_frac=0.5):
    M = np.zeros((bins,bins), dtype=bool)
    s = int(np.round(bins*(1-center_frac)/2))
    M[s:bins-s, s:bins-s] = True
    return M

def plot_coverage(pos_df, bins=40, center_frac=0.5):
    occ, xe, ye = occupancy_map(pos_df, bins)
    M = center_mask(occ.shape[0], center_frac)
    center_time = float(occ[M].sum())
    boundary_time = float(occ[~M].sum())

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(occ, origin="lower", extent=[0,ARENA_CM,0,ARENA_CM])
    plt.title("Occupancy (s)"); plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(M, origin="lower", extent=[0,ARENA_CM,0,ARENA_CM], alpha=0.35)
    plt.title(f"Center mask (center={center_frac*100:.0f}%); "
              f"center={center_time:.1f}s, boundary={boundary_time:.1f}s")
    plt.show()

def rate_map(pos_df, spike_ts, bins=40, sigma=1.0):
    occ, xe, ye = occupancy_map(pos_df, bins)
    k = pos_df[pos_df["keep"]]
    t = k["t"].to_numpy(); x = k["x"].to_numpy(); y = k["y"].to_numpy()
    idx = np.searchsorted(t, spike_ts); idx = np.clip(idx, 0, len(t)-1)
    sx, sy = x[idx], y[idx]
    S, _, _ = np.histogram2d(sx, sy, bins=[xe, ye]); S = S.T
    S_s = gaussian_filter(S, sigma=sigma, mode="nearest")
    O_s = gaussian_filter(occ, sigma=sigma, mode="nearest")
    with np.errstate(divide='ignore', invalid='ignore'):
        R = np.where(O_s>0, S_s/O_s, np.nan)
    return R

def plot_rate_map(R, title="Rate map (Hz)"):
    plt.figure(figsize=(5,5))
    plt.imshow(R, origin="lower", extent=[0,ARENA_CM,0,ARENA_CM])
    plt.title(title); plt.colorbar(); plt.show()

def path_plus_spikes(pos_df, spike_ts):
    k = pos_df[pos_df["keep"]]
    t = k["t"].to_numpy(); x = k["x"].to_numpy(); y = k["y"].to_numpy()
    idx = np.searchsorted(t, spike_ts); idx = np.clip(idx, 0, len(t)-1)
    sx, sy = x[idx], y[idx]
    plt.figure(figsize=(6,6))
    plt.plot(x, y, lw=0.3, alpha=0.5, label="path")
    plt.scatter(sx, sy, s=6, alpha=0.9, label="spikes")
    ax = plt.gca(); ax.set_aspect("equal"); ax.set(xlim=(0,ARENA_CM), ylim=(0,ARENA_CM))
    plt.legend(); plt.title("Spatial coverage (path + spikes)")
    plt.show()
