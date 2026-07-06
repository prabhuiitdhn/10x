"""
Step 1: Explore the ROS2 bag file (.db3) schema and raw message data.
ROS2 bags store data in SQLite3 with tables: topics, messages.
Messages are serialized using CDR (Common Data Representation).
"""

import sqlite3
import struct

DB_PATH = r'd:\innovation\10x\depth\depth.db3'

con = sqlite3.connect(DB_PATH)
cur = con.cursor()

# ── 1. List all tables ────────────────────────────────────────────────────────
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cur.fetchall()
print("Tables in bag:", [t[0] for t in tables])

# ── 2. Show schema of each table ──────────────────────────────────────────────
for (t,) in tables:
    cur.execute(f"PRAGMA table_info({t})")
    cols = cur.fetchall()
    print(f"\n--- {t} ---")
    for c in cols:
        print(f"  {c}")

# ── 3. Show topics ────────────────────────────────────────────────────────────
print("\n\n=== TOPICS ===")
cur.execute("SELECT * FROM topics")
for row in cur.fetchall():
    print(row)

# ── 4. Show message count and timestamps ──────────────────────────────────────
print("\n\n=== MESSAGES (id, topic_id, timestamp, data_size) ===")
cur.execute("SELECT id, topic_id, timestamp, length(data) FROM messages ORDER BY timestamp")
for row in cur.fetchall():
    print(f"  msg_id={row[0]}  topic_id={row[1]}  ts={row[2]}  data_bytes={row[3]}")

# ── 5. Peek into first message CDR header ─────────────────────────────────────
# sensor_msgs/msg/Image CDR layout (after 4-byte CDR header):
#   std_msgs/Header (stamp: int32 sec + uint32 nanosec, frame_id: string)
#   uint32 height, uint32 width
#   string encoding
#   uint8  is_bigendian
#   uint32 step (row stride in bytes)
#   uint8[] data
print("\n\n=== FIRST MESSAGE RAW PARSE ===")
cur.execute("SELECT data FROM messages ORDER BY timestamp LIMIT 1")
raw = cur.fetchone()[0]
offset = 4  # skip CDR encapsulation header (4 bytes)

# Header.stamp
sec,  = struct.unpack_from('<i', raw, offset);  offset += 4
nsec, = struct.unpack_from('<I', raw, offset);  offset += 4
# Header.frame_id (pascal-style: uint32 length then bytes)
fid_len, = struct.unpack_from('<I', raw, offset); offset += 4
frame_id = raw[offset:offset+fid_len].decode('utf-8', errors='replace').rstrip('\x00')
offset += fid_len
# Align to 4 bytes
if offset % 4: offset += 4 - (offset % 4)

height, = struct.unpack_from('<I', raw, offset); offset += 4
width,  = struct.unpack_from('<I', raw, offset); offset += 4

# encoding string
enc_len, = struct.unpack_from('<I', raw, offset); offset += 4
encoding = raw[offset:offset+enc_len].decode('utf-8', errors='replace').rstrip('\x00')
offset += enc_len
if offset % 4: offset += 4 - (offset % 4)

is_bigendian = struct.unpack_from('<B', raw, offset)[0]; offset += 1
if offset % 4: offset += 4 - (offset % 4)

step, = struct.unpack_from('<I', raw, offset); offset += 4
data_len, = struct.unpack_from('<I', raw, offset); offset += 4

print(f"  stamp      : {sec}.{nsec:09d} s")
print(f"  frame_id   : {frame_id!r}")
print(f"  height     : {height}")
print(f"  width      : {width}")
print(f"  encoding   : {encoding!r}")
print(f"  is_bigendian: {is_bigendian}")
print(f"  step       : {step}  (bytes per row)")
print(f"  data_len   : {data_len}  (total pixel bytes)")
print(f"  expected   : {height * step}  (height*step)")

con.close()
