from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

from brainrender import Scene
from brainrender.actors import Points
from brainrender import Scene, settings


settings.SHOW_AXES = False
# — you can list as many folders as you like here —
results_folders = [
   Path(r"C:\Users\xinxin.yin\OneDrive - Allen Institute\probe_reconstruction\after_registration_tracks"),
   # Path(r"C:\Users\xinxin.yin\OneDrive - Allen Institute\probe_reconstruction\after_registration_tracks_before"),
    #Path(r"C:\Users\xinxin.yin\OneDrive - Allen Institute\probe_reconstruction\New folder\results"),
    # add more Path(...) entries as needed
]

# 1) Gather all .fcsv files from every folder
all_files = []
for folder in results_folders:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Results folder not found: {folder}")
    all_files.extend(folder.glob("*.fcsv"))

# 2) Group by probe key (strip '_fit' and '_ShankN')
groups = {}
for f in all_files:
    name = f.name
    if name.endswith("_fit.fcsv"):
        base = name[:-len("_fit.fcsv")]
    else:
        base = name[:-len(".fcsv")]
    probe_key = re.sub(r"_Shank\d+$", "", base)
    groups.setdefault(probe_key, []).append(f)

# 3) Prepare Brainrender scene
scene = Scene(title="18 sessions in 7 mice")
scene.add_brain_region("MD", alpha=0.15)

# 4) Choose a colormap and convert to hex strings
cmap    = plt.get_cmap("tab10")
hexcols = [to_hex(cmap(i)) for i in range(cmap.N)]

# 5) Loop over each probe group, assign a single color, add all its tracks
for idx, (probe_key, files) in enumerate(sorted(groups.items())):
    color = hexcols[idx % len(hexcols)]
    print(f"Probe {probe_key!r}: {len(files)} file(s) → color {color}")
    for fcsv in files:
        df = pd.read_csv(
            fcsv, comment="#", header=None,
            names=[
                "id","x","y","z","ow","ox","oy","oz",
                "vis","sel","lock","label","desc","associatedNodeID"
            ],
        )
        pts_lps = df[["x","y","z"]].to_numpy(dtype=float)

        # Convert LPS → PVL and scale mm→µm
        pts_pvl = np.column_stack([
            pts_lps[:, 1] * 1000,   # scene-X = Posterior
           -pts_lps[:, 2] * 1000,   # scene-Y = Ventral
           -pts_lps[:, 0] * 1000,   # scene-Z = Left
        ])

        scene.add(
            Points(
                pts_pvl,
                name=fcsv.stem,  # e.g. 'ProbeB_Shank4' or 'ProbeB_Shank4_fit'
                colors=color,
                radius=15,
            )
        )

# 6) Render the complete scene
scene.render()
