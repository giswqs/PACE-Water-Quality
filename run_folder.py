"""Process PACE L2 AOP scenes in a folder into per-scene water-quality COGs.

Every PACE pass in the folder is run through the models and written as
validated Cloud Optimized GeoTIFFs (chl-a, TSS, aCDOM). Each COG keeps the
granule name of its input and lives in a per-product subfolder, so dates with
several passes keep every pass::

    output/chla/PACE_OCI.20240929T185124.L2.OC_AOP.V3_2.tif
    output/tss/PACE_OCI.20240929T185124.L2.OC_AOP.V3_2.tif
    output/acdom/PACE_OCI.20240929T185124.L2.OC_AOP.V3_2.tif

Products are gridded from the swath onto a regular EPSG:4326 grid. Scenes
whose COGs already exist are skipped unless ``--overwrite`` is passed, so an
interrupted backfill can simply be re-run.

Examples::

    python run_folder.py                       # process ./data -> ./output
    python run_folder.py /media/hdd/Data/PACE/data --output /media/hdd/Data/PACE/output
    python run_folder.py --json-dir /media/hdd/Data/PACE/json

To process a single file, use ``run_file.py``.
"""

import argparse
import glob
import os

import torch

from pace_processing import (
    BASE_DIR,
    infer_scene_maps,
    load_models,
    parse_acquisition_date,
    scene_cog_paths,
    scene_stem,
    write_scene_cogs,
)

parser = argparse.ArgumentParser(
    description="Process every PACE scene in a folder into per-scene "
    "water-quality COGs."
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
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Reprocess scenes whose COGs already exist (default: skip them).",
)
parser.add_argument(
    "--json-dir",
    default=None,
    help="If set, write the per-date GeoJSON catalogs here after processing.",
)
parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Process at most this many scenes (useful for a quick test run).",
)
args = parser.parse_args()

if not os.path.isdir(args.folder):
    raise NotADirectoryError(f"Input folder not found: {args.folder}")

# Collect input scenes (skip any products files written by older runs).
scenes = [
    path
    for path in sorted(glob.glob(os.path.join(args.folder, args.pattern)))
    if not path.endswith("_products.nc")
]

if not scenes:
    raise FileNotFoundError(
        f"No files matching '{args.pattern}' found in {args.folder}"
    )

# Skip scenes that already have a complete set of COGs.
if not args.overwrite:
    pending = [
        path
        for path in scenes
        if not all(
            os.path.isfile(p)
            for p in scene_cog_paths(args.output, scene_stem(path)).values()
        )
    ]
    n_skipped = len(scenes) - len(pending)
    if n_skipped:
        print(f"Skipping {n_skipped} scene(s) that already have COGs.")
    scenes = pending

if args.limit is not None:
    scenes = scenes[: args.limit]

if not scenes:
    print("Nothing to process.")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dates = {parse_acquisition_date(p) for p in scenes}
    print(f"Processing {len(scenes)} scene(s) across {len(dates)} date(s)")

    models = load_models(args.model_dir, device)

    succeeded, failed = [], []
    for i, nc_path in enumerate(scenes, start=1):
        name = os.path.basename(nc_path)
        print(f"\n[{i}/{len(scenes)}] {name}")
        try:
            maps = infer_scene_maps(nc_path, models)
            print(f"  {maps['valid']} valid pixels")
            write_scene_cogs(maps, args.output, scene_stem(nc_path))
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            print(f"  FAILED: {exc}")
            failed.append((nc_path, exc))
            continue
        succeeded.append(nc_path)

    print(f"\nDone. {len(succeeded)} scene(s) written, {len(failed)} failed.")
    for nc_path, exc in failed:
        print(f"  - {os.path.basename(nc_path)}: {exc}")

if args.json_dir:
    from make_json import build_catalogs

    build_catalogs(args.output, args.json_dir)
