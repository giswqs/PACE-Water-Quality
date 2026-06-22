"""Process every PACE L2 AOP scene in a folder into water-quality products.

Loads the MoE-VAE models once and runs inference on each PACE NetCDF file in
the input folder, writing the products NetCDF plus validated Cloud Optimized
GeoTIFFs (chl-a, TSS, aCDOM) to the output folder. Scenes that fail are
reported and skipped so one bad file does not abort the whole batch.

Examples::

    python run_folder.py                       # process ./data -> ./output
    python run_folder.py /path/to/scenes
    python run_folder.py data --output results --pattern "PACE_OCI.*V3_2.nc"

To process a single file, use ``run_file.py``.
"""

import os
import glob
import argparse

import torch

from pace_processing import BASE_DIR, load_models, process_scene

parser = argparse.ArgumentParser(
    description="Process every PACE L2 AOP scene in a folder."
)
parser.add_argument(
    "folder",
    nargs="?",
    default=os.path.join(BASE_DIR, "data"),
    help="Folder containing PACE NetCDF files (default: ./data).",
)
parser.add_argument(
    "--output",
    default=os.path.join(BASE_DIR, "output"),
    help="Output directory for the products (default: ./output).",
)
parser.add_argument(
    "--model-dir",
    default=os.path.join(BASE_DIR, "model"),
    help="Directory containing the model subfolders (default: ./model).",
)
parser.add_argument(
    "--pattern",
    default="*.nc",
    help="Glob pattern for input files (default: *.nc).",
)
args = parser.parse_args()

if not os.path.isdir(args.folder):
    raise NotADirectoryError(f"Input folder not found: {args.folder}")

# Match the pattern but never treat already-written products as input.
scenes = sorted(
    f
    for f in glob.glob(os.path.join(args.folder, args.pattern))
    if not f.endswith("_products.nc")
)
if not scenes:
    raise FileNotFoundError(
        f"No files matching '{args.pattern}' found in {args.folder}"
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Found {len(scenes)} scene(s) in {args.folder}")

models = load_models(args.model_dir, device)

succeeded, failed = [], []
for i, nc_path in enumerate(scenes, 1):
    print(f"\n[{i}/{len(scenes)}] {os.path.basename(nc_path)}")
    try:
        process_scene(nc_path, models, args.output)
        succeeded.append(nc_path)
    except Exception as exc:  # noqa: BLE001 - keep batch going on failure
        print(f"  FAILED: {exc}")
        failed.append((nc_path, exc))

print(f"\nDone. {len(succeeded)} succeeded, {len(failed)} failed.")
for nc_path, exc in failed:
    print(f"  - {os.path.basename(nc_path)}: {exc}")
