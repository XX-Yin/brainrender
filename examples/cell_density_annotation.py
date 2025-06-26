import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from brainrender import Scene, settings
from brainrender.actors import Points

settings.SHOW_AXES = False

# -----------------------------
# USER PARAMETERS
# -----------------------------
alignment_root = r"C:\Users\xinxin.yin\OneDrive - Allen Institute\probe_reconstruction\units_annotated\results"
brain_region_filter = ["MD"]  # set to [] or None to disable filtering

# -----------------------------
# FIND ALL SESSION FILES
# -----------------------------
all_files = [
    fname for fname in os.listdir(alignment_root)
    if ((fname.startswith("ccf_channel_") and fname.endswith(".json"))
        or fname.endswith("_ccf_loc.json"))
]
if not all_files:
    raise RuntimeError("No JSON files matched your naming patterns.")

# prepare colors
cmap = plt.get_cmap("tab10")
hex_colors = [to_hex(cmap(i % cmap.N)) for i in range(len(all_files))]

# -----------------------------
# PROCESS AND COLLECT SESSION DATA
# -----------------------------
session_data = []     # to hold tuples of (session_name, pts_pvl, hex_color)
session_counts = {}   # to hold counts per session

for fname, hex_color in zip(all_files, hex_colors):
    fpath = os.path.join(alignment_root, fname)
    try:
        with open(fpath, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"⚠️ Skipping {fname}: {e}")
        continue

    pts_lps = []
    for channel_info in data.values():
        if not isinstance(channel_info, dict):
            continue
        region = channel_info.get("brain_region")
        x, y, z = channel_info.get("x"), channel_info.get("y"), channel_info.get("z")
        if brain_region_filter and region not in brain_region_filter:
            continue
        if x is None or y is None or z is None:
            continue
        pts_lps.append([x, y, z])

    count = len(pts_lps)
    session_name = os.path.splitext(fname)[0]
    session_counts[session_name] = count

    if count == 0:
        continue

    pts_lps = np.array(pts_lps)
    pts_pvl = np.column_stack([
        pts_lps[:, 1] * 1000,    # Posterior → X (µm)
        -pts_lps[:, 2] * 1000,   # Superior → Y (µm) (negated)
        -pts_lps[:, 0] * 1000,   # Left     → Z (µm) (negated)
    ])

    session_data.append((session_name, pts_pvl, hex_color))

# -----------------------------
# COMPUTE TOTAL AND SESSION COUNT
# -----------------------------
total_units = sum(session_counts.values())
num_sessions = len(session_counts)

# -----------------------------
# SET UP AND RENDER SCENE
# -----------------------------
scene = Scene(title=f"{total_units} units in MD ({num_sessions} sessions in 5 mice)")
scene.add_brain_region("MD", alpha=0.15)

for session_name, pts_pvl, hex_color in session_data:
    scene.add(
        Points(
            pts_pvl,
            name=session_name,
            colors=hex_color,
            radius=20
        )
    )

scene.render()

# -----------------------------
# PRINT STATISTICS
# -----------------------------
counts = np.array(list(session_counts.values()))

print(f"Processed {num_sessions} sessions.")
print(f"Total units across all sessions: {total_units}")
print(f"Units per session – min: {counts.min()}, max: {counts.max()}, "
      f"mean: {counts.mean():.1f}, median: {np.median(counts)}\n")

print("Units by session:")
for session, cnt in session_counts.items():
    print(f"  {session}: {cnt}")
