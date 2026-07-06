"""
Step 5: Estimate the axis of rotation.

Key insight:
  When a rigid body rotates about a fixed axis, the face normal traces a circle
  on the unit sphere whose plane is perpendicular to the rotation axis.
  Therefore the rotation axis is the normal to the best-fit plane through all
  the face-normal unit vectors.

Algorithm:
  1. Collect the 7 unit face-normals from Step 3.
  2. PCA on those normals: the eigenvector with the SMALLEST eigenvalue
     (least variance direction) is perpendicular to the plane of the circle
     → that is the rotation axis.
  3. Align the axis sign so it points "up" in camera space (positive Y
     convention) or positive Z — whichever is more meaningful.
  4. Sanity-check: project each normal onto the plane perpendicular to the
     axis and verify the angular progression is monotone.
"""

import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

OUT_DIR = Path(r'd:\innovation\10x\output')

# ── Load per-frame normals ────────────────────────────────────────────────────
with open(str(OUT_DIR / 'per_frame_results.json')) as f:
    results = json.load(f)

normals = np.array([r['normal'] for r in results])   # (7, 3)
times   = np.array([r['time_s'] for r in results])   # (7,)

print("Face normals (unit vectors):")
for i, (n, t) in enumerate(zip(normals, times)):
    print(f"  Frame {i+1}  t={t:.3f}s  n={np.round(n, 4)}")

# Verify they are roughly unit
norms = np.linalg.norm(normals, axis=1)
print(f"\nNorm range: {norms.min():.4f} – {norms.max():.4f}  (should be ≈1)")

# ── PCA to find rotation axis ─────────────────────────────────────────────────
# Centre the normals (they should already lie on the unit sphere, but centering
# removes any systematic offset and is standard PCA practice).
centroid = normals.mean(axis=0)
centered = normals - centroid

cov = centered.T @ centered            # 3×3 covariance-like matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov)   # ascending eigenvalues

# The smallest eigenvalue corresponds to the axis of rotation
rot_axis = eigenvectors[:, 0]          # column 0 = smallest eigenvalue
rot_axis = rot_axis / np.linalg.norm(rot_axis)

# Convention: axis should have a positive Y component (points "upward" in
# camera frame) — flip if needed
if rot_axis[1] < 0:
    rot_axis = -rot_axis

print(f"\nEigenvalues (ascending): {np.round(eigenvalues, 6)}")
print(f"Rotation axis (camera frame): {np.round(rot_axis, 6)}")

# ── Sanity check: angular progression ────────────────────────────────────────
# Project each normal onto the plane perpendicular to rot_axis, then measure
# the angle of the projection (i.e. the rotation angle of the box at each frame).

# Build an orthonormal basis in the plane ⊥ to rot_axis
u = np.array([1.0, 0.0, 0.0])
u -= u.dot(rot_axis) * rot_axis
u /= np.linalg.norm(u)
v = np.cross(rot_axis, u)

angles_rad = []
for n in normals:
    n_proj = n - n.dot(rot_axis) * rot_axis   # project onto plane
    n_proj /= max(np.linalg.norm(n_proj), 1e-9)
    theta = np.arctan2(n_proj.dot(v), n_proj.dot(u))
    angles_rad.append(theta)

angles_deg = np.degrees(angles_rad)
print("\nBox rotation angle per frame (relative to Frame 1):")
angles_deg_rel = angles_deg - angles_deg[0]
for i, (a, t) in enumerate(zip(angles_deg_rel, times)):
    print(f"  Frame {i+1}  t={t:.3f}s  rotation={a:.2f}°")

# ── 3-D visualisation ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection='3d')

# Draw unit sphere wireframe
u_s = np.linspace(0, 2*np.pi, 30)
v_s = np.linspace(0, np.pi, 20)
xs = np.outer(np.cos(u_s), np.sin(v_s))
ys = np.outer(np.sin(u_s), np.sin(v_s))
zs = np.outer(np.ones(len(u_s)), np.cos(v_s))
ax.plot_wireframe(xs, ys, zs, alpha=0.05, color='gray')

# Draw normals as dots + connecting arc
ax.scatter(normals[:, 0], normals[:, 1], normals[:, 2],
           c=times, cmap='viridis', s=60, zorder=5, label='Face normals')
for i, (n, t) in enumerate(zip(normals, times)):
    ax.text(n[0]*1.05, n[1]*1.05, n[2]*1.05, f'F{i+1}', fontsize=7)

# Draw rotation axis
ax.quiver(0, 0, 0, rot_axis[0], rot_axis[1], rot_axis[2],
          length=1.2, color='red', linewidth=2, label=f'Rot axis')

ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('Face normals on unit sphere\n(rotation axis = red arrow)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(str(OUT_DIR / 'step5_rotation_axis.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── Save rotation axis to text file ──────────────────────────────────────────
axis_txt = OUT_DIR / 'rotation_axis.txt'
with open(str(axis_txt), 'w') as f:
    f.write("Axis of rotation (camera frame)\n")
    f.write("=================================\n")
    f.write(f"X = {rot_axis[0]:.6f}\n")
    f.write(f"Y = {rot_axis[1]:.6f}\n")
    f.write(f"Z = {rot_axis[2]:.6f}\n")
    f.write(f"\nUnit vector: [{rot_axis[0]:.6f}, {rot_axis[1]:.6f}, {rot_axis[2]:.6f}]\n")
    f.write(f"\nNote: Estimated via PCA on per-frame face normals.\n")
    f.write(f"The smallest-eigenvalue eigenvector of the normal covariance matrix\n")
    f.write(f"is perpendicular to the plane traced by the rotating face normal,\n")
    f.write(f"and therefore equals the rotation axis.\n")

print(f"\nSaved: rotation_axis.txt  step5_rotation_axis.png")
