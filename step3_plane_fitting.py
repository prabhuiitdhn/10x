"""
Step 3: For each depth frame —
  1. Back-project valid pixels into 3-D camera space (X, Y, Z).
  2. Isolate the box region (foreground = pixels closer than a background threshold).
  3. Use RANSAC plane fitting to find the dominant flat face of the box.
  4. From the fitted plane normal, compute:
       - Normal angle  : angle between the face normal and the camera optical axis (Z-axis)
       - Visible area  : number of inlier pixels × per-pixel area (m²)
  5. Collect per-frame results and save for Step 4/5.

Camera intrinsics assumption:
  The bag has no /camera_info topic, so we use typical RealSense D435 defaults
  for a 640×480 depth stream.  All results scale correctly as long as the
  aspect ratio and FOV are representative.
"""

import numpy as np
from pathlib import Path
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = Path(r'd:\innovation\10x\output')

frames     = np.load(str(OUT_DIR / 'frames.npy'))        # (7, 480, 640)  metres
timestamps = np.load(str(OUT_DIR / 'timestamps.npy'))    # (7,) ns

H, W = frames.shape[1], frames.shape[2]

# ── Camera intrinsics (RealSense D435 @ 640×480, typical defaults) ────────────
# fx = fy ≈ 381 px  (horizontal FOV ~87°, vertical ~58°)
fx = 381.0
fy = 381.0
cx = W / 2.0   # 320.0
cy = H / 2.0   # 240.0

def depth_to_pointcloud(depth_m):
    """Back-project a 480×640 depth image to (N,3) XYZ points in camera frame.
    Camera frame convention: Z forward, X right, Y down.
    Only valid (non-NaN) pixels are returned.
    Returns: points (N,3), (row, col) indices of valid pixels.
    """
    rows, cols = np.where(~np.isnan(depth_m))
    z = depth_m[rows, cols]
    x = (cols - cx) * z / fx
    y = (rows - cy) * z / fy
    return np.column_stack([x, y, z]), rows, cols

def ransac_plane(points, n_iter=200, dist_thresh=0.02):
    """Fit a plane ax+by+cz+d=0 to points using RANSAC.
    Returns: normal (unit, 3), d, inlier_mask (bool, N)
    """
    best_inliers = None
    best_count   = 0
    N = len(points)

    for _ in range(n_iter):
        idx = np.random.choice(N, 3, replace=False)
        p0, p1, p2 = points[idx]
        v1 = p1 - p0
        v2 = p2 - p0
        n  = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -np.dot(n, p0)
        dist = np.abs(points @ n + d)
        inliers = dist < dist_thresh
        cnt = inliers.sum()
        if cnt > best_count:
            best_count   = cnt
            best_inliers = inliers

    # Refine with all inliers (SVD least-squares, economy form)
    if best_inliers is not None and best_inliers.sum() >= 3:
        pts = points[best_inliers]
        centroid = pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
        n = Vt[-1]           # normal = last singular vector
        n = n / np.linalg.norm(n)
        d = -np.dot(n, centroid)
        dist = np.abs(points @ n + d)
        best_inliers = dist < dist_thresh

    return n, d, best_inliers

# Camera optical axis (Z forward)
Z_AXIS = np.array([0.0, 0.0, 1.0])

results = []   # list of dicts per frame

fig, axes = plt.subplots(2, len(frames), figsize=(3 * len(frames), 7))
fig.suptitle('Step 3 — Box Segmentation & Plane Fitting', fontsize=13)

for i, (depth, ts) in enumerate(zip(frames, timestamps)):
    t_sec = (ts - timestamps[0]) * 1e-9

    # ── 1. Back-project ───────────────────────────────────────────────────────
    pts, rows, cols = depth_to_pointcloud(depth)

    # ── 2. Foreground segmentation ────────────────────────────────────────────
    # The box is the nearest large object.  Use the 15th-percentile depth as a
    # rough near-distance; keep pixels within 1.5× of it.
    z_vals = pts[:, 2]
    z_near = np.percentile(z_vals, 15)
    fg_mask = z_vals < (z_near * 1.8)
    fg_pts  = pts[fg_mask]

    print(f"Frame {i+1}: total_pts={len(pts):,}  fg_pts={len(fg_pts):,}  "
          f"z_near={z_near:.3f}m")

    # ── 3. RANSAC plane fitting ───────────────────────────────────────────────
    normal, d, inliers = ransac_plane(fg_pts, n_iter=300, dist_thresh=0.025)

    # Ensure normal points toward camera (positive Z component)
    if normal[2] < 0:
        normal = -normal
        d = -d

    inlier_pts = fg_pts[inliers]
    n_inliers  = inliers.sum()

    # ── 4. Normal angle (degrees between face normal and camera Z-axis) ───────
    cos_theta = np.clip(np.dot(normal, Z_AXIS), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_theta))

    # ── 5. Visible area (m²) ─────────────────────────────────────────────────
    # Project inlier points onto the plane's local 2-D coordinate system, then
    # compute the convex-hull area as the visible face area.
    # Local basis: u = first right-singular vector of inlier pts, v = u × normal
    _, _, Vt = np.linalg.svd(inlier_pts - inlier_pts.mean(axis=0), full_matrices=False)
    u_axis = Vt[0]
    v_axis = np.cross(normal, u_axis)
    proj_u = inlier_pts @ u_axis
    proj_v = inlier_pts @ v_axis
    proj_2d = np.column_stack([proj_u, proj_v])

    try:
        hull = ConvexHull(proj_2d)
        area_m2 = hull.volume   # in 2-D, ConvexHull.volume = area
    except Exception:
        area_m2 = float('nan')

    results.append({
        'frame'     : i + 1,
        'time_s'    : round(t_sec, 3),
        'normal'    : normal.tolist(),
        'angle_deg' : round(angle_deg, 2),
        'area_m2'   : round(area_m2, 4),
        'n_inliers' : int(n_inliers),
    })

    print(f"         normal={np.round(normal,4)}  angle={angle_deg:.2f}°  "
          f"area={area_m2:.4f}m²  inliers={n_inliers:,}")

    # ── Visualise ─────────────────────────────────────────────────────────────
    seg_img = np.zeros((H, W), dtype=np.float32)
    fg_rows = rows[fg_mask]
    fg_cols = cols[fg_mask]
    seg_img[fg_rows[inliers], fg_cols[inliers]] = 2   # plane inliers (bright)
    seg_img[fg_rows[~inliers], fg_cols[~inliers]] = 1 # other foreground

    axes[0, i].imshow(np.clip(depth, 0, 5), cmap='plasma', vmin=0, vmax=5)
    axes[0, i].set_title(f'F{i+1} depth\n+{t_sec:.1f}s', fontsize=8)
    axes[0, i].axis('off')

    axes[1, i].imshow(seg_img, cmap='viridis', vmin=0, vmax=2)
    axes[1, i].set_title(f'{angle_deg:.1f}°  {area_m2:.3f}m²', fontsize=8)
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig(str(OUT_DIR / 'step3_segmentation.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── Save results ──────────────────────────────────────────────────────────────
import json
with open(str(OUT_DIR / 'per_frame_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\n=== PER-FRAME RESULTS ===")
print(f"{'Frame':>5}  {'Time(s)':>8}  {'Angle(°)':>9}  {'Area(m²)':>9}  {'Inliers':>8}")
print("-" * 50)
for r in results:
    print(f"{r['frame']:>5}  {r['time_s']:>8.3f}  {r['angle_deg']:>9.2f}  "
          f"{r['area_m2']:>9.4f}  {r['n_inliers']:>8}")

print("\nSaved: step3_segmentation.png  per_frame_results.json")
