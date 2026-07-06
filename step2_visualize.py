"""
Step 2: Visualise all 7 depth frames.
- Top row: raw depth map (colorised)
- Bottom row: valid-pixel mask (NaN = no-return regions)
This helps us see how the box is rotating across frames.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')           # non-interactive backend (saves to file)
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(r'd:\innovation\10x\output')

frames     = np.load(str(OUT_DIR / 'frames.npy'))        # (7, 480, 640)
timestamps = np.load(str(OUT_DIR / 'timestamps.npy'))    # (7,) ns

n = len(frames)
fig, axes = plt.subplots(2, n, figsize=(3 * n, 7))
fig.suptitle('Depth Frames — Rotating Cuboid', fontsize=14)

for i, (depth, ts) in enumerate(zip(frames, timestamps)):
    t_sec = (ts - timestamps[0]) * 1e-9   # relative time in seconds

    # --- depth image (clip to 0–5 m for visibility) ---
    ax = axes[0, i]
    im = ax.imshow(np.clip(depth, 0, 5), cmap='plasma', vmin=0, vmax=5)
    ax.set_title(f'Frame {i+1}\n+{t_sec:.2f}s', fontsize=8)
    ax.axis('off')

    # --- valid mask ---
    ax2 = axes[1, i]
    ax2.imshow(~np.isnan(depth), cmap='gray')
    valid_pct = 100 * np.sum(~np.isnan(depth)) / depth.size
    ax2.set_title(f'Valid {valid_pct:.1f}%', fontsize=8)
    ax2.axis('off')

# shared colorbar
cbar_ax = fig.add_axes([0.92, 0.55, 0.015, 0.35])
sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, 5))
plt.colorbar(sm, cax=cbar_ax, label='Depth (m)')

plt.tight_layout(rect=[0, 0, 0.92, 1])
out_png = OUT_DIR / 'frames_overview.png'
plt.savefig(str(out_png), dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {out_png}")

# ── Print depth statistics per frame ─────────────────────────────────────────
print(f"\n{'Frame':>5}  {'Time(s)':>8}  {'MinD(m)':>8}  {'MaxD(m)':>8}  "
      f"{'MeanD(m)':>9}  {'ValidPx':>8}")
print("-" * 60)
for i, (depth, ts) in enumerate(zip(frames, timestamps)):
    t = (ts - timestamps[0]) * 1e-9
    v = depth[~np.isnan(depth)]
    print(f"{i+1:>5}  {t:>8.3f}  {v.min():>8.3f}  {v.max():>8.3f}  "
          f"{v.mean():>9.3f}  {v.size:>8}")
