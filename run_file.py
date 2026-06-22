"""Process a single PACE L2 AOP scene into water-quality products.

Runs MoE-VAE inference on one PACE NetCDF file and writes the products
NetCDF plus validated Cloud Optimized GeoTIFFs (chl-a, TSS, aCDOM) to the
local ``output`` folder.

Examples::

    python run_file.py PACE_OCI.20240929T185124.L2.OC_AOP.V3_0.nc
    python run_file.py data/PACE_OCI.20240701T175112.L2.OC_AOP.V3_1.nc
    python run_file.py /path/to/scene.nc --output /path/to/output

To process every scene in a folder, use ``run_folder.py``.
"""

import os
import argparse

import torch

from pace_processing import BASE_DIR, load_models, process_scene

parser = argparse.ArgumentParser(
    description="Process a single PACE L2 AOP scene into water-quality products."
)
parser.add_argument(
    "input",
    help="Input PACE NetCDF file. Either a path, or a filename in the "
    "'data' folder.",
)
parser.add_argument(
    "--model-dir",
    default=os.path.join(BASE_DIR, "model"),
    help="Directory containing the model subfolders (default: ./model).",
)
parser.add_argument(
    "--output",
    default=os.path.join(BASE_DIR, "output"),
    help="Output directory for the products (default: ./output).",
)
args = parser.parse_args()

# Resolve the input path: use it as given if it exists, otherwise look in
# the local 'data' folder next to this script.
if os.path.isfile(args.input):
    nc_path = os.path.abspath(args.input)
else:
    nc_path = os.path.join(BASE_DIR, "data", args.input)
if not os.path.isfile(nc_path):
    raise FileNotFoundError(f"Input file not found: {args.input}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

models = load_models(args.model_dir, device)
process_scene(nc_path, models, args.output)
print("Done.")
