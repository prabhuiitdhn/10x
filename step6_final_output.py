"""
Final output generator.
Produces:
  1. results_table.txt  — image number, normal angle, visible area
  2. rotation_axis.txt  — already written by step5, confirmed here
  3. Console summary
"""

import json
from pathlib import Path

OUT_DIR = Path(r'd:\innovation\10x\output')

with open(str(OUT_DIR / 'per_frame_results.json')) as f:
    results = json.load(f)

# Read axis
with open(str(OUT_DIR / 'rotation_axis.txt')) as f:
    axis_content = f.read()

# ── Write results table ───────────────────────────────────────────────────────
table_path = OUT_DIR / 'results_table.txt'
with open(str(table_path), 'w') as f:
    f.write("Results Table — Perception Assignment\n")
    f.write("======================================\n\n")
    f.write(f"{'Image No.':<12}{'Timestamp (s)':<16}{'Normal Angle (°)':<20}{'Visible Area (m²)':<20}\n")
    f.write("-" * 68 + "\n")
    for r in results:
        f.write(f"{r['frame']:<12}{r['time_s']:<16.3f}{r['angle_deg']:<20.2f}{r['area_m2']:<20.4f}\n")
    f.write("\n")
    f.write("Notes:\n")
    f.write("  Normal Angle : angle between the largest visible face normal\n")
    f.write("                 and the camera optical axis (Z-axis), in degrees.\n")
    f.write("  Visible Area : convex-hull area of the RANSAC plane inliers\n")
    f.write("                 projected onto the face plane, in m².\n")

print(f"Saved: {table_path}")
print(f"Saved: {OUT_DIR / 'rotation_axis.txt'}")
print()
print("=" * 68)
print("RESULTS TABLE")
print("=" * 68)
print(f"{'Image No.':<12}{'Timestamp (s)':<16}{'Normal Angle (°)':<20}{'Visible Area (m²)'}")
print("-" * 68)
for r in results:
    print(f"{r['frame']:<12}{r['time_s']:<16.3f}{r['angle_deg']:<20.2f}{r['area_m2']:.4f}")

print()
print("=" * 68)
print("AXIS OF ROTATION (camera frame)")
print("=" * 68)
print(axis_content)
