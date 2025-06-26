import json
from pathlib import Path
from itertools import chain
import numpy as np
from brainrender import Scene
from brainrender.actors import Points, PointsDensity

# -----------------------------
# USER PARAMETERS
# -----------------------------
alignment_root = Path(
    r"C:\Users\xinxin.yin\OneDrive - Allen Institute\probe_reconstruction\units_annotated\results"
)


alignment_root = Path(
    r"C:\Users\xinxin.yin\OneDrive - Allen Institute"
    r"\probe_reconstruction\alignment"
)

brain_region_filter = ["MD", "PVT"]  # set to [] or None to disable filtering

# -----------------------------
# COLLECT AND FILTER COORDINATES
# -----------------------------
pts_lps = []  # will hold [x, y, z] in mm

for result_folder in alignment_root.iterdir():
    if not result_folder.is_dir():
        continue
    for session_folder in result_folder.iterdir():
        if not (session_folder.is_dir() and session_folder.name.startswith("ecephys_")):
            continue
        for probe_folder in session_folder.iterdir():
            if not (probe_folder.is_dir() and probe_folder.name.startswith("Probe")):
                continue

            # look for both naming conventions
            for jf in chain(
                probe_folder.glob("ccf_channel_*.json"),
                probe_folder.glob("*_ccf_loc.json"),
            ):
                # load JSON, skip if unreadable
                try:
                    data = json.loads(jf.read_text())
                except (json.JSONDecodeError, OSError) as e:
                    print(f"  ⚠️ Skipping {jf.name}: {e}")
                    continue

                # iterate over channels
                for channel_key, channel_info in data.items():
                    if not isinstance(channel_info, dict):
                        continue

                    # get fields safely
                    region = channel_info.get("brain_region")
                    x = channel_info.get("x")
                    y = channel_info.get("y")
                    z = channel_info.get("z")
                    # apply filter and skip missing values
                    if brain_region_filter and region not in brain_region_filter:
                        continue
                    if x is None or y is None or z is None:
                        continue

                    pts_lps.append([x, y, z])

# convert to numpy array (N×3) in mm
pts_lps = np.array(pts_lps)

# -----------------------------
# TRANSFORM TO BRAINRENDER SPACE
# -----------------------------
pts_pvl = np.column_stack([
    pts_lps[:, 1] * 1000,   # Posterior → scene X (µm)
    -pts_lps[:, 2] * 1000,  # Superior → scene Y (µm) (negated for ventral)
    -pts_lps[:, 0] * 1000,  # Left      → scene Z (µm)
])

# -----------------------------
# VISUALIZE
# -----------------------------
scene = Scene(title="Filtered Probes (PVL, µm)")
scene.add_brain_region("MD", alpha=0.15)
scene.add(Points(pts_pvl, name="Neurons", colors="salmon"))
# scene.add(PointsDensity(pts_pvl))
scene.render()
