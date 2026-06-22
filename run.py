import torch
import numpy as np
import os
import rasterio
import sys
import pickle
import argparse
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rio_cogeo.cogeo import cog_translate, cog_validate
from rio_cogeo.profiles import cog_profiles

# Resolve paths relative to this script so it can run from any location.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "code"))
from MoE_VAE import *
from data_loading import  *
from plot_and_save import *
from model_inference import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Parameters ===
selected_bands = [
    400, 403, 405, 408, 410, 413, 415, 418, 420, 422, 425, 427, 430, 432, 435,
    437, 440, 442, 445, 447, 450, 452, 455, 457, 460, 462, 465, 467, 470, 472,
    475, 477, 480, 482, 485, 487, 490, 492, 495, 497, 500, 502, 505, 507, 510,
    512, 515, 517, 520, 522, 525, 527, 530, 532, 535, 537, 540, 542, 545, 547,
    550, 553, 555, 558, 560, 563, 565, 568, 570, 573, 575, 578, 580, 583, 586,
    588, 613, 615, 618, 620, 623, 625, 627, 630, 632, 635, 637, 640, 641, 642,
    643, 645, 646, 647, 648, 650, 651, 652, 653, 655, 656, 657, 658, 660, 661,
    662, 663, 665, 666, 667, 668, 670, 671, 672, 673, 675, 676, 677, 678, 679,
    681, 682, 683, 684, 686, 687, 688, 689, 691, 692, 693, 694, 696, 697, 698,
    699, 701, 702, 703, 704, 706, 707, 708, 709, 711, 712, 713, 714, 717, 719]

# === Command-line arguments ===
parser = argparse.ArgumentParser(
    description="Run MoE-VAE inference on a PACE L2 AOP scene and write "
    "products (NetCDF + Cloud Optimized GeoTIFFs)."
)
parser.add_argument(
    "input",
    nargs="?",
    default="PACE_OCI.20240929T185124.L2.OC_AOP.V3_0.nc",
    help="Input PACE NetCDF file. Either a path, or a filename that lives in "
    "the 'data' folder. Defaults to the bundled test scene.",
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

model_dir = os.path.join(BASE_DIR, "model")
save_dir = os.path.join(BASE_DIR, "output")
os.makedirs(save_dir, exist_ok=True)
print(f"Input scene: {nc_path}")
# ============================
# Chl-a
# ============================

chla_model = MoE_VAE(
    input_dim=len(selected_bands),
    output_dim=1,
    latent_dim=32,
    encoder_hidden_dims=[64, 64],
    decoder_hidden_dims=[64, 64],
    activation='leakyrelu',
    use_norm='layer',
    use_dropout=False,
    use_softplus_output=True,
    num_experts=4,
    k=2,
    noisy_gating=True
).to(device)

chla_model.load_state_dict(
    torch.load(
        os.path.join(model_dir, "chl-a", "best_model_minloss.pth"),
        map_location=device
    )
)

chla_output = preprocess_infer_pace_minmax(
    nc_path=nc_path,
    model=chla_model,
    full_band_wavelengths=selected_bands,
    use_spectral_mask=True,
    batch_size=2048,
    log_offset=1
)

# ============================
# TSS
# ============================

tss_model = MoE_VAE(
    input_dim=len(selected_bands),
    output_dim=1,
    latent_dim=16,
    encoder_hidden_dims=[64, 32],
    decoder_hidden_dims=[32, 64],
    activation='leakyrelu',
    use_norm='layer',
    use_dropout=False,
    use_softplus_output=False,
    num_experts=4,
    k=2,
    noisy_gating=True
).to(device)

tss_model.load_state_dict(
    torch.load(
        os.path.join(model_dir, "tss", "best_model_minloss.pth"),
        map_location=device
    )
)

with open(
    os.path.join(model_dir, "tss", "scalers_Rrs_real.pkl"),
    "rb"
) as f:
    tss_scalers_Rrs = pickle.load(f)

tss_scalers_dict = torch.load(
    os.path.join(model_dir, "tss", "scaler.pt"),
    map_location="cpu",
    weights_only=False
)

tss_output = preprocess_infer_pace_robust(
    nc_path=nc_path,
    model=tss_model,
    scaler_Rrs=tss_scalers_Rrs,
    TSS_scalers_dict=tss_scalers_dict,
    full_band_wavelengths=selected_bands,
    use_diff=False,
    use_spectral_mask=True,
    batch_size=2048
)

# ============================
# aCDOM
# ============================

acdom_model = MoE_VAE(
    input_dim=len(selected_bands),
    output_dim=1,
    latent_dim=32,
    encoder_hidden_dims=[256, 128, 64],
    decoder_hidden_dims=[64, 128, 256],
    activation='leakyrelu',
    use_norm='layer',
    use_dropout=False,
    use_softplus_output=False,
    num_experts=4,
    k=2,
    noisy_gating=True
).to(device)

acdom_model.load_state_dict(
    torch.load(
        os.path.join(model_dir, "acdom", "best_model_minloss.pth"),
        map_location=device
    )
)

with open(
    os.path.join(model_dir, "acdom", "scalers_Rrs_real.pkl"),
    "rb"
) as f:
    acdom_scalers_Rrs = pickle.load(f)

acdom_scalers_dict = torch.load(
    os.path.join(model_dir, "acdom", "scaler.pt"),
    map_location="cpu",
    weights_only=False
)

acdom_output = preprocess_infer_pace_robust(
    nc_path=nc_path,
    model=acdom_model,
    scaler_Rrs=acdom_scalers_Rrs,
    TSS_scalers_dict=acdom_scalers_dict,
    full_band_wavelengths=selected_bands,
    use_diff=False,
    use_spectral_mask=True,
    batch_size=2048
)
products_nc = save_pace_products_to_nc(
    nc_path=nc_path,
    save_dir=save_dir,
    chla_output=chla_output,
    tss_output=tss_output,
    acdom_output=acdom_output
)


def save_product_to_cog(
    out_tif,
    lat_2d,
    lon_2d,
    values_2d,
    resolution=None,
    nodata=-9999.0,
    footprint_factor=1.5,
):
    """Grid a PACE swath product onto a regular grid and write a COG.

    The PACE L2 product is a swath with 2D (curvilinear) latitude/longitude.
    Following HyperCoast's ``grid_pace`` approach, valid pixels are
    resampled onto a regular EPSG:4326 grid with nearest-neighbour
    interpolation (``scipy.interpolate.griddata``), which avoids the
    moiré/striping that simple binning produces when the target grid is
    finer than the native pixel spacing. Because the retrieval is a sparse
    coastal footprint (clear-water pixels between clouds), grid cells whose
    nearest valid pixel is farther than ``footprint_factor`` cells away are
    set to nodata so values are not smeared across open water/land. The
    result is written as a Cloud Optimized GeoTIFF with internal tiling,
    overviews and DEFLATE compression, then validated.

    Args:
        out_tif (str): Output GeoTIFF path.
        lat_2d (np.ndarray): 2D latitude array (degrees north, EPSG:4326).
        lon_2d (np.ndarray): 2D longitude array (degrees east, EPSG:4326).
        values_2d (np.ndarray): 2D product values aligned with lat/lon.
        resolution (float, optional): Grid cell size in degrees. Defaults to
            the median native pixel spacing (so output ≈ native resolution).
        nodata (float): Value used for empty cells.
        footprint_factor (float): Cells farther than this many grid cells
            from the nearest valid pixel are masked to nodata.

    Returns:
        str: The path to the validated COG.
    """
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree

    lat = np.asarray(lat_2d, dtype=np.float64).ravel()
    lon = np.asarray(lon_2d, dtype=np.float64).ravel()
    val = np.asarray(values_2d, dtype=np.float64).ravel()

    mask = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(val)
    lat, lon, val = lat[mask], lon[mask], val[mask]
    if lat.size == 0:
        raise ValueError(f"No valid pixels to grid for {out_tif}")

    points = np.column_stack([lon, lat])

    # Native pixel spacing from median nearest-neighbour distance.
    if resolution is None:
        tree = cKDTree(points)
        nn = tree.query(points, k=2)[0][:, 1]
        resolution = float(np.median(nn)) * 1.1

    lon0 = lon.min() - resolution / 2.0
    lon1 = lon.max() + resolution / 2.0
    lat0 = lat.min() - resolution / 2.0
    lat1 = lat.max() + resolution / 2.0
    ncol = int(np.ceil((lon1 - lon0) / resolution))
    nrow = int(np.ceil((lat1 - lat0) / resolution))
    transform = from_origin(lon0, lat1, resolution, resolution)

    # Regular grid cell centres; row 0 = north (lat descending).
    grid_lon = lon0 + (np.arange(ncol) + 0.5) * resolution
    grid_lat = lat1 - (np.arange(nrow) + 0.5) * resolution
    mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)

    # Nearest-neighbour resampling (HyperCoast grid_pace technique).
    gridded = griddata(points, val, (mesh_lon, mesh_lat), method="nearest")

    # Mask cells outside the actual data footprint to avoid smearing.
    tree = cKDTree(points)
    dist = tree.query(np.column_stack([mesh_lon.ravel(), mesh_lat.ravel()]))[0]
    dist = dist.reshape(mesh_lon.shape)
    inside = dist <= resolution * footprint_factor

    grid = np.full((nrow, ncol), nodata, dtype=np.float32)
    grid[inside] = gridded[inside].astype(np.float32)

    src_profile = dict(
        driver="GTiff",
        dtype="float32",
        count=1,
        height=nrow,
        width=ncol,
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
    )
    dst_profile = cog_profiles.get("deflate")
    with MemoryFile() as mem:
        with mem.open(**src_profile) as src:
            src.write(grid, 1)
        with mem.open() as src:
            cog_translate(
                src,
                out_tif,
                dst_profile,
                overview_resampling="nearest",
                quiet=True,
            )

    valid, errors, warnings = cog_validate(out_tif)
    status = "valid" if valid else "INVALID"
    print(
        f"COG {status}: {out_tif} "
        f"({int(inside.sum())} cells @ {resolution:.4f} deg)"
    )
    if errors:
        print("  errors:", errors)
    if warnings:
        print("  warnings:", warnings)
    return out_tif


# ============================
# Cloud Optimized GeoTIFFs
# ============================
import re
import xarray as xr

# Parse the acquisition date (YYYYMMDD) from the input filename, e.g.
# "PACE_OCI.20240929T185124.L2.OC_AOP.V3_0.nc" -> "20240929".
match = re.search(r"(\d{8})T\d{6}", os.path.basename(nc_path))
if match is None:
    raise ValueError(
        f"Could not parse acquisition date from filename: {nc_path}"
    )
acq_date = match.group(1)

products_ds = xr.open_dataset(products_nc)
lat_2d = products_ds["latitude"].values
lon_2d = products_ds["longitude"].values
# Map dataset variable -> output filename label.
product_labels = {"chla": "chla", "tss": "tss", "acdom440": "acdom"}
for var, label in product_labels.items():
    save_product_to_cog(
        out_tif=os.path.join(save_dir, f"PACE_OCI-{acq_date}-{label}.tif"),
        lat_2d=lat_2d,
        lon_2d=lon_2d,
        values_2d=products_ds[var].values,
    )
products_ds.close()