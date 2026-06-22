"""Composite multiple same-day PACE passes into one daily product per date.

There are typically several PACE overpasses per day over a region, each
covering part of the swath (and often broken up by clouds). This script
pools the valid pixels from every ``*_products.nc`` sharing an acquisition
date and grids them together, producing a single Cloud Optimized GeoTIFF per
date per product (``PACE_OCI-YYYYMMDD-<label>.tif``) with the best combined
coverage for that day.

It reads the intermediate ``*_products.nc`` files written by the inference
step (so run the processing with ``keep_nc=True`` / before they are deleted).

Examples::

    python composite_folder.py /media/hdd/Data/PACE/output
    python composite_folder.py /media/hdd/Data/PACE/output --output composites
"""

import os
import re
import glob
import argparse
from collections import defaultdict

import numpy as np
import xarray as xr

from pace_processing import save_product_to_cog, PRODUCT_LABELS

parser = argparse.ArgumentParser(
    description="Composite same-day PACE passes into one daily product per date."
)
parser.add_argument(
    "folder",
    help="Folder containing the intermediate *_products.nc files.",
)
parser.add_argument(
    "--output",
    default=None,
    help="Output directory for the daily composite COGs (default: same folder).",
)
parser.add_argument(
    "--pattern",
    default="*_products.nc",
    help="Glob pattern for the products NetCDF files.",
)
args = parser.parse_args()

out_dir = args.output or args.folder
os.makedirs(out_dir, exist_ok=True)

# Group products files by acquisition date (YYYYMMDD).
by_date = defaultdict(list)
for path in sorted(glob.glob(os.path.join(args.folder, args.pattern))):
    m = re.search(r"(\d{8})T\d{6}", os.path.basename(path))
    if m:
        by_date[m.group(1)].append(path)

if not by_date:
    raise FileNotFoundError(
        f"No files matching '{args.pattern}' found in {args.folder}"
    )

print(f"Found {sum(len(v) for v in by_date.values())} passes across "
      f"{len(by_date)} dates.")

for date in sorted(by_date):
    passes = by_date[date]
    print(f"\n[{date}] compositing {len(passes)} pass(es)")

    # Pool valid pixels (lat, lon, value) across all passes for each product.
    pooled = {var: {"lat": [], "lon": [], "val": []} for var in PRODUCT_LABELS}
    for path in passes:
        ds = xr.open_dataset(path)
        lat = ds["latitude"].values.ravel()
        lon = ds["longitude"].values.ravel()
        for var in PRODUCT_LABELS:
            val = ds[var].values.ravel()
            mask = np.isfinite(val) & np.isfinite(lat) & np.isfinite(lon)
            pooled[var]["lat"].append(lat[mask])
            pooled[var]["lon"].append(lon[mask])
            pooled[var]["val"].append(val[mask])
        ds.close()

    for var, label in PRODUCT_LABELS.items():
        lat = np.concatenate(pooled[var]["lat"])
        lon = np.concatenate(pooled[var]["lon"])
        val = np.concatenate(pooled[var]["val"])
        save_product_to_cog(
            out_tif=os.path.join(out_dir, f"PACE_OCI-{date}-{label}.tif"),
            lat_2d=lat,
            lon_2d=lon,
            values_2d=val,
        )

print("\nDone compositing daily products.")
