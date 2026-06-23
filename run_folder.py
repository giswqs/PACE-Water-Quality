"""Process PACE L2 AOP scenes in a folder into daily water-quality COGs.

For each acquisition date, every PACE pass in the folder is run through the
models and the pass with the most valid retrieval pixels is kept, written as
date-named Cloud Optimized GeoTIFFs (chl-a, TSS, aCDOM) in the output folder.
Products are written directly from the swath (no gridding/interpolation), so
each pixel keeps its exact model value.

Examples::

    python run_folder.py                       # process ./data -> ./output
    python run_folder.py /media/hdd/Data/PACE/data --output /media/hdd/Data/PACE/output

To process a single file, use ``run_file.py``.
"""

import os
import glob
import argparse
from collections import defaultdict

import torch

from pace_processing import (
    BASE_DIR,
    load_models,
    infer_scene_maps,
    write_scene_cogs,
    parse_acquisition_date,
)

parser = argparse.ArgumentParser(
    description="Process PACE scenes in a folder into daily water-quality COGs "
    "(best pass per day)."
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

# Group input scenes by acquisition date (skip any products files).
by_date = defaultdict(list)
for path in sorted(glob.glob(os.path.join(args.folder, args.pattern))):
    if path.endswith("_products.nc"):
        continue
    by_date[parse_acquisition_date(path)].append(path)

if not by_date:
    raise FileNotFoundError(
        f"No files matching '{args.pattern}' found in {args.folder}"
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
n_scenes = sum(len(v) for v in by_date.values())
print(f"Found {n_scenes} scene(s) across {len(by_date)} date(s) in {args.folder}")

models = load_models(args.model_dir, device)

succeeded, failed = [], []
for date in sorted(by_date):
    passes = by_date[date]
    print(f"\n[{date}] {len(passes)} pass(es)")
    best_maps, best_pass = None, None
    for nc_path in passes:
        try:
            maps = infer_scene_maps(nc_path, models)
        except Exception as exc:  # noqa: BLE001 - keep batch going on failure
            print(f"  FAILED {os.path.basename(nc_path)}: {exc}")
            failed.append((nc_path, exc))
            continue
        print(f"  {os.path.basename(nc_path)}: {maps['valid']} valid pixels")
        if best_maps is None or maps["valid"] > best_maps["valid"]:
            best_maps, best_pass = maps, nc_path

    if best_maps is None:
        continue
    print(f"  -> best: {os.path.basename(best_pass)} ({best_maps['valid']} px)")
    write_scene_cogs(best_maps, args.output, date)
    succeeded.append(date)

print(f"\nDone. {len(succeeded)} date(s) written, {len(failed)} pass(es) failed.")
for nc_path, exc in failed:
    print(f"  - {os.path.basename(nc_path)}: {exc}")
