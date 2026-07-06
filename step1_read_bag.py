"""
Step 1 (revised): Read the ROS2 bag using the `rosbags` library.
sensor_msgs/msg/Image with encoding '16UC1':
  - Each pixel is a uint16 value (distance in millimetres).
  - We convert to float metres for all further processing.
"""

import numpy as np
from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

BAG_DIR = Path(r'd:\innovation\10x\depth')
OUT_DIR = Path(r'd:\innovation\10x\output')
OUT_DIR.mkdir(exist_ok=True)

typestore = get_typestore(Stores.ROS2_HUMBLE)

frames = []   # list of (timestamp_ns, depth_array_metres)

with Reader(BAG_DIR) as reader:
    connections = [c for c in reader.connections if c.topic == '/depth']
    print(f"Found {len(connections)} connection(s) on /depth")

    for connection, timestamp, rawdata in reader.messages(connections=connections):
        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

        h, w = msg.height, msg.width
        enc  = msg.encoding          # expect '16UC1'
        step = msg.step              # bytes per row

        print(f"  ts={timestamp}  shape=({h},{w})  enc={enc}  step={step}")

        # Reconstruct pixel array
        raw_bytes = bytes(msg.data)
        if enc == '16UC1':
            arr = np.frombuffer(raw_bytes, dtype=np.uint16).reshape(h, w)
            # ROS depth convention: 0 = invalid/no-return; unit = mm
            depth_m = arr.astype(np.float32) / 1000.0   # convert mm → metres
            depth_m[arr == 0] = np.nan                   # mark invalid pixels
        elif enc == '32FC1':
            arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape(h, w)
            depth_m = arr.copy()
            depth_m[~np.isfinite(depth_m)] = np.nan
        else:
            raise ValueError(f"Unexpected encoding: {enc}")

        frames.append((timestamp, depth_m))

print(f"\nTotal frames extracted: {len(frames)}")

# Quick sanity check: depth range of first frame
ts0, d0 = frames[0]
valid = d0[~np.isnan(d0)]
print(f"Frame 0  min={valid.min():.3f}m  max={valid.max():.3f}m  "
      f"mean={valid.mean():.3f}m  valid_pixels={valid.size}/{d0.size}")

# Save frames as numpy arrays for next steps
np.save(str(OUT_DIR / 'frames.npy'),
        np.array([f[1] for f in frames]))           # shape (7, 480, 640)
np.save(str(OUT_DIR / 'timestamps.npy'),
        np.array([f[0] for f in frames]))           # shape (7,)

print("\nSaved frames.npy and timestamps.npy to output/")
