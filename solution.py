"""
=================================
Task : A 3D cuboidal box rotates about its central axis in front of a depth
       camera.  Estimate per-frame face-normal angle & visible area, and the
       global axis of rotation.

Input  : ROS2 bag  depth/depth.db3   (7 × 480×640, encoding 16UC1, mm)
Outputs: output/results_table.txt
         output/rotation_axis.txt
         output/frames_overview.png
         output/result_plot.png
"""

import numpy as np
import json
import matplotlib;

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import ConvexHull
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

# path is being defined for input and output
BAG_DIR = Path('depth')
OUT_DIR = Path('output');
OUT_DIR.mkdir(exist_ok=True)

# assumed camera parameter
FX, FY = 381.0, 381.0  # focal lengths (pixels)
CX, CY = 320.0, 240.0  # principal point (pixels)

# RANSAC parameters; going to use this for estimating with less data set.
# traditional cv technique for estimating the value inlcuding outliers

RANSAC_ITERS = 300  # number of random trials
RANSAC_THRESH = 0.025  # inlier threshold

# Foreground segmentation: keep pixels within this factor of the nearest depth
FG_FACTOR = 1.8

# STEP 1 — READ BAG
print("STEP 1: Reading ROS2 bag")

typestore = get_typestore(Stores.ROS2_HUMBLE)
frames, timestamps = [], []

with Reader(BAG_DIR) as reader:
    conns = [c for c in reader.connections if c.topic == '/depth']
    for conn, ts, raw in reader.messages(connections=conns):
        msg = typestore.deserialize_cdr(raw, conn.msgtype)

        # 16UC1 → float metres  (0 means invalid → NaN)
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(msg.height, msg.width)
        depth = arr.astype(np.float32) / 1000.0
        depth[arr == 0] = np.nan

        frames.append(depth);
        timestamps.append(ts)
        print(f"  frame {len(frames)}  ts={ts}  shape={depth.shape}")

frames = np.array(frames)  # (7, 480, 640)
timestamps = np.array(timestamps)  # (7,)  nanoseconds
print(f"Total frames: {len(frames)}\n")


# STEP 2 — BACK-PROJECT DEPTH → 3-D POINTS  (pin-hole camera model)
# For each pixel (u, v) with depth Z:
#   X = (u - cx) * Z / fx          (right)
#   Y = (v - cy) * Z / fy          (down)
#   Z = depth value                 (forward)

def backproject(depth):
    """Return (N,3) XYZ array for all valid pixels."""
    v_idx, u_idx = np.where(~np.isnan(depth))
    Z = depth[v_idx, u_idx]
    X = (u_idx - CX) * Z / FX
    Y = (v_idx - CY) * Z / FY
    return np.column_stack([X, Y, Z]), v_idx, u_idx


# STEP 3 — RANSAC PLANE FIT
# A plane is defined by  n·p + d = 0  where n is the unit normal.
# RANSAC robustly finds the largest set of points lying on one flat face.

def ransac_plane(pts):
    """Fit a plane to pts (N,3) using RANSAC. Returns normal, d, inlier_mask."""
    N = len(pts)
    best_mask = np.zeros(N, dtype=bool)

    for _ in range(RANSAC_ITERS):
        i0, i1, i2 = np.random.choice(N, 3, replace=False)
        n = np.cross(pts[i1] - pts[i0], pts[i2] - pts[i0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = -n.dot(pts[i0])
        mask = np.abs(pts @ n + d) < RANSAC_THRESH
        if mask.sum() > best_mask.sum():
            best_mask = mask

    # Refine: refit using all inliers (SVD least-squares)
    inliers = pts[best_mask]
    centroid = inliers.mean(axis=0)
    _, _, Vt = np.linalg.svd(inliers - centroid, full_matrices=False)
    n = Vt[-1] / np.linalg.norm(Vt[-1])
    d = -n.dot(centroid)

    # Final inlier mask with refined normal
    best_mask = np.abs(pts @ n + d) < RANSAC_THRESH
    return n, d, best_mask


# STEP 4 — PER-FRAME: NORMAL ANGLE + VISIBLE AREA
# Normal angle : arccos(n · Z_hat)  where Z_hat = [0,0,1] (camera optical axis)
# Visible area : convex-hull area of the inlier points projected onto the face plane

Z_AXIS = np.array([0.0, 0.0, 1.0])

print("STEP 4: Per-frame plane fitting")
print(f"{'Frame':>5}  {'Time(s)':>7}  {'Angle(°)':>9}  {'Area(m²)':>9}")

per_frame = []  # collects results for output

fig_vis, axes = plt.subplots(2, len(frames), figsize=(3 * len(frames), 6))
fig_vis.suptitle('Depth frames & box segmentation', fontsize=12)

for i, (depth, ts) in enumerate(zip(frames, timestamps)):
    t = (ts - timestamps[0]) * 1e-9  # elapsed time in seconds

    # --- back-project ---
    pts, vv, uu = backproject(depth)

    # --- isolate box (foreground = nearest cluster) ---
    z_near = np.percentile(pts[:, 2], 15)
    fg = pts[:, 2] < z_near * FG_FACTOR
    fg_pts = pts[fg]

    # --- fit plane ---
    normal, d, inliers = ransac_plane(fg_pts)
    if normal[2] < 0:  # ensure normal faces the camera
        normal, d = -normal, -d

    inlier_pts = fg_pts[inliers]

    # --- normal angle ---
    angle_deg = np.degrees(np.arccos(np.clip(normal.dot(Z_AXIS), -1, 1)))

    # --- visible area via convex hull in face's local 2-D frame ---
    _, _, Vt2 = np.linalg.svd(inlier_pts - inlier_pts.mean(0), full_matrices=False)
    u_ax = Vt2[0];
    v_ax = np.cross(normal, u_ax)
    proj = np.column_stack([inlier_pts @ u_ax, inlier_pts @ v_ax])
    try:
        area_m2 = ConvexHull(proj).volume  # .volume == area in 2-D
    except Exception:
        area_m2 = float('nan')

    per_frame.append({'frame': i + 1, 'time_s': round(t, 3),
                      'normal': normal.tolist(),
                      'angle_deg': round(angle_deg, 2),
                      'area_m2': round(area_m2, 4)})

    print(f"{i + 1:>5}  {t:>7.3f}  {angle_deg:>9.2f}  {area_m2:>9.4f}")

    # visualise
    seg = np.zeros(depth.shape, dtype=np.float32)
    seg[vv[fg][inliers], uu[fg][inliers]] = 2  # plane inliers
    seg[vv[fg][~inliers], uu[fg][~inliers]] = 1  # other foreground
    axes[0, i].imshow(np.clip(depth, 0, 5), cmap='plasma', vmin=0, vmax=5)
    axes[0, i].set_title(f'F{i + 1}  +{t:.1f}s', fontsize=8);
    axes[0, i].axis('off')
    axes[1, i].imshow(seg, cmap='viridis', vmin=0, vmax=2)
    axes[1, i].set_title(f'{angle_deg:.1f}°  {area_m2:.3f}m²', fontsize=8)
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig(str(OUT_DIR / 'result_plot.png'), dpi=120, bbox_inches='tight')
plt.close()

# STEP 5 — AXIS OF ROTATION  (PCA on face normals)

# When a rigid body rotates about a fixed axis, its face normal traces a circle on the unit sphere.
# The rotation axis is perpendicular to that circle's plane.
# PCA gives us that: the eigenvector with the SMALLEST eigenvalue of the normal
# covariance matrix is the rotation axis.


print("STEP 5: Estimating axis of rotation")

normals = np.array([r['normal'] for r in per_frame])  # (7, 3)
centroid = normals.mean(axis=0)
cov = (normals - centroid).T @ (normals - centroid)
evals, evecs = np.linalg.eigh(cov)  # ascending eigenvalues
rot_axis = evecs[:, 0]  # smallest eigenvalue → axis
rot_axis /= np.linalg.norm(rot_axis)
if rot_axis[1] < 0:  # sign convention
    rot_axis = -rot_axis

print(f"Eigenvalues : {np.round(evals, 5)}")
print(f"Rotation axis (camera frame): {np.round(rot_axis, 6)}")

# STEP 6 — WRITE OUTPUTS

# --- results_table.txt ---
with open(OUT_DIR / 'results_table.txt', 'w') as f:
    f.write("Results Table\n")
    f.write("=" * 60 + "\n")
    f.write(f"{'Image No.':<12}{'Time (s)':<12}{'Normal Angle (°)':<20}{'Visible Area (m²)'}\n")
    f.write("-" * 60 + "\n")
    for r in per_frame:
        f.write(f"{r['frame']:<12}{r['time_s']:<12.3f}{r['angle_deg']:<20.2f}{r['area_m2']:.4f}\n")

# --- rotation_axis.txt ---
with open(OUT_DIR / 'rotation_axis.txt', 'w') as f:
    f.write("Axis of Rotation (camera frame)\n")
    f.write("=" * 40 + "\n")
    f.write(f"X = {rot_axis[0]:.6f}\n")
    f.write(f"Y = {rot_axis[1]:.6f}\n")
    f.write(f"Z = {rot_axis[2]:.6f}\n")
    f.write(f"\nUnit vector: {np.round(rot_axis, 6).tolist()}\n")

print("FINAL RESULTS TABLE")
print(f"{'Image No.':<12}{'Time (s)':<12}{'Normal Angle (°)':<20}{'Visible Area (m²)'}")

for r in per_frame:
    print(f"{r['frame']:<12}{r['time_s']:<12.3f}{r['angle_deg']:<20.2f}{r['area_m2']:.4f}")

print(f"\nRotation axis (camera frame): {np.round(rot_axis, 6).tolist()}")
