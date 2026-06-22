"""Shared PACE water-quality processing logic.

This module holds the reusable pieces of the MoE-VAE inference workflow so
that the model weights only have to be loaded once and can be reused across
many scenes:

* :func:`load_models` - build the chl-a / TSS / aCDOM models and scalers.
* :func:`process_scene` - run inference on a single PACE L2 AOP scene and
  write the products NetCDF plus validated Cloud Optimized GeoTIFFs.
* :func:`save_product_to_cog` - grid a swath product and write a valid COG.

Entry points:

* ``run_file.py`` processes a single file.
* ``run_folder.py`` processes every scene in a folder.
"""

import os
import re
import sys
import pickle

import numpy as np
import torch
import xarray as xr
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rio_cogeo.cogeo import cog_translate, cog_validate
from rio_cogeo.profiles import cog_profiles

# Resolve paths relative to this module so it can run from any location.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "code"))

from MoE_VAE import *  # noqa: E402,F401,F403
from data_loading import *  # noqa: E402,F401,F403
from plot_and_save import *  # noqa: E402,F401,F403
from model_inference import *  # noqa: E402,F401,F403

# Wavelengths (nm) used as model input features.
SELECTED_BANDS = [
    400,
    403,
    405,
    408,
    410,
    413,
    415,
    418,
    420,
    422,
    425,
    427,
    430,
    432,
    435,
    437,
    440,
    442,
    445,
    447,
    450,
    452,
    455,
    457,
    460,
    462,
    465,
    467,
    470,
    472,
    475,
    477,
    480,
    482,
    485,
    487,
    490,
    492,
    495,
    497,
    500,
    502,
    505,
    507,
    510,
    512,
    515,
    517,
    520,
    522,
    525,
    527,
    530,
    532,
    535,
    537,
    540,
    542,
    545,
    547,
    550,
    553,
    555,
    558,
    560,
    563,
    565,
    568,
    570,
    573,
    575,
    578,
    580,
    583,
    586,
    588,
    613,
    615,
    618,
    620,
    623,
    625,
    627,
    630,
    632,
    635,
    637,
    640,
    641,
    642,
    643,
    645,
    646,
    647,
    648,
    650,
    651,
    652,
    653,
    655,
    656,
    657,
    658,
    660,
    661,
    662,
    663,
    665,
    666,
    667,
    668,
    670,
    671,
    672,
    673,
    675,
    676,
    677,
    678,
    679,
    681,
    682,
    683,
    684,
    686,
    687,
    688,
    689,
    691,
    692,
    693,
    694,
    696,
    697,
    698,
    699,
    701,
    702,
    703,
    704,
    706,
    707,
    708,
    709,
    711,
    712,
    713,
    714,
    717,
    719,
]

# Map dataset variable -> output filename label (aCDOM drops the "440").
PRODUCT_LABELS = {"chla": "chla", "tss": "tss", "acdom440": "acdom"}


def load_models(model_dir, device):
    """Build the chl-a, TSS and aCDOM models and load their weights/scalers.

    Args:
        model_dir (str): Directory containing the ``chl-a``, ``tss`` and
            ``acdom`` model subfolders.
        device (torch.device): Device to load the models onto.

    Returns:
        dict: Mapping of product name to a dict with the loaded ``model`` and,
            for TSS/aCDOM, the ``scaler_Rrs`` and ``scaler_dict`` objects.
    """
    n_bands = len(SELECTED_BANDS)

    chla_model = MoE_VAE(
        input_dim=n_bands,
        output_dim=1,
        latent_dim=32,
        encoder_hidden_dims=[64, 64],
        decoder_hidden_dims=[64, 64],
        activation="leakyrelu",
        use_norm="layer",
        use_dropout=False,
        use_softplus_output=True,
        num_experts=4,
        k=2,
        noisy_gating=True,
    ).to(device)
    chla_model.load_state_dict(
        torch.load(
            os.path.join(model_dir, "chl-a", "best_model_minloss.pth"),
            map_location=device,
        )
    )

    tss_model = MoE_VAE(
        input_dim=n_bands,
        output_dim=1,
        latent_dim=16,
        encoder_hidden_dims=[64, 32],
        decoder_hidden_dims=[32, 64],
        activation="leakyrelu",
        use_norm="layer",
        use_dropout=False,
        use_softplus_output=False,
        num_experts=4,
        k=2,
        noisy_gating=True,
    ).to(device)
    tss_model.load_state_dict(
        torch.load(
            os.path.join(model_dir, "tss", "best_model_minloss.pth"),
            map_location=device,
        )
    )
    with open(os.path.join(model_dir, "tss", "scalers_Rrs_real.pkl"), "rb") as f:
        tss_scaler_Rrs = pickle.load(f)
    tss_scaler_dict = torch.load(
        os.path.join(model_dir, "tss", "scaler.pt"),
        map_location="cpu",
        weights_only=False,
    )

    acdom_model = MoE_VAE(
        input_dim=n_bands,
        output_dim=1,
        latent_dim=32,
        encoder_hidden_dims=[256, 128, 64],
        decoder_hidden_dims=[64, 128, 256],
        activation="leakyrelu",
        use_norm="layer",
        use_dropout=False,
        use_softplus_output=False,
        num_experts=4,
        k=2,
        noisy_gating=True,
    ).to(device)
    acdom_model.load_state_dict(
        torch.load(
            os.path.join(model_dir, "acdom", "best_model_minloss.pth"),
            map_location=device,
        )
    )
    with open(os.path.join(model_dir, "acdom", "scalers_Rrs_real.pkl"), "rb") as f:
        acdom_scaler_Rrs = pickle.load(f)
    acdom_scaler_dict = torch.load(
        os.path.join(model_dir, "acdom", "scaler.pt"),
        map_location="cpu",
        weights_only=False,
    )

    return {
        "chla": {"model": chla_model},
        "tss": {
            "model": tss_model,
            "scaler_Rrs": tss_scaler_Rrs,
            "scaler_dict": tss_scaler_dict,
        },
        "acdom": {
            "model": acdom_model,
            "scaler_Rrs": acdom_scaler_Rrs,
            "scaler_dict": acdom_scaler_dict,
        },
    }


def save_product_to_cog(
    out_tif,
    lat_2d,
    lon_2d,
    values_2d,
    resolution=None,
    nodata=-9999.0,
    footprint_factor=1.5,
    fill_gaps=True,
    max_gap_cells=5,
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
    set to nodata so values are not smeared across open water/land.

    With ``fill_gaps`` enabled, interior holes (clouds/flagged pixels/scan
    gaps that are *enclosed* by valid data) are filled: small gaps are
    bridged with a morphological closing and fully-enclosed holes are filled,
    then the nearest-neighbour values are pulled into those cells. The true
    outer boundary of the footprint is left as nodata, so open ocean/land is
    not filled. The result is written as a Cloud Optimized GeoTIFF with
    internal tiling, overviews and DEFLATE compression, then validated.

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
        fill_gaps (bool): If True, fill interior holes enclosed by data.
        max_gap_cells (int): Bridge nodata gaps up to roughly this many cells
            wide before filling enclosed holes.

    Returns:
        str: The path to the validated COG.
    """
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    from scipy.ndimage import (
        binary_closing,
        binary_fill_holes,
        generate_binary_structure,
    )

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

    # Fill interior holes (clouds/flagged pixels/scan gaps) enclosed by data,
    # without expanding the true outer boundary into open ocean/land.
    if fill_gaps:
        struct = generate_binary_structure(2, 2)  # 8-connectivity
        coverage = binary_closing(inside, structure=struct, iterations=max_gap_cells)
        coverage = binary_fill_holes(coverage)
        # Closing can bleed past the boundary; keep only cells that still have
        # a nearby valid pixel, plus the newly enclosed interior holes.
        inside = coverage & (dist <= resolution * (footprint_factor + max_gap_cells))

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


def parse_acquisition_date(nc_path):
    """Parse the acquisition date (YYYYMMDD) from a PACE filename.

    Args:
        nc_path (str): Path to the PACE NetCDF file, e.g.
            ``PACE_OCI.20240929T185124.L2.OC_AOP.V3_0.nc``.

    Returns:
        str: The 8-digit date string (e.g. ``"20240929"``).

    Raises:
        ValueError: If no date can be parsed from the filename.
    """
    match = re.search(r"(\d{8})T\d{6}", os.path.basename(nc_path))
    if match is None:
        raise ValueError(f"Could not parse acquisition date from: {nc_path}")
    return match.group(1)


def process_scene(nc_path, models, save_dir, keep_nc=False):
    """Run inference on one PACE scene and write the NetCDF + COG products.

    Args:
        nc_path (str): Path to the input PACE L2 AOP NetCDF file.
        models (dict): Loaded models/scalers from :func:`load_models`.
        save_dir (str): Directory to write the products into.
        keep_nc (bool): If False (default), the intermediate products NetCDF
            is deleted once the GeoTIFFs are written. The COGs are the
            deliverable, so this avoids accumulating large NetCDF files.

    Returns:
        list[str]: Paths to the written COG files.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"Processing scene: {nc_path}")

    chla_output = preprocess_infer_pace_minmax(
        nc_path=nc_path,
        model=models["chla"]["model"],
        full_band_wavelengths=SELECTED_BANDS,
        use_spectral_mask=True,
        batch_size=2048,
        log_offset=1,
    )

    tss_output = preprocess_infer_pace_robust(
        nc_path=nc_path,
        model=models["tss"]["model"],
        scaler_Rrs=models["tss"]["scaler_Rrs"],
        TSS_scalers_dict=models["tss"]["scaler_dict"],
        full_band_wavelengths=SELECTED_BANDS,
        use_diff=False,
        use_spectral_mask=True,
        batch_size=2048,
    )

    acdom_output = preprocess_infer_pace_robust(
        nc_path=nc_path,
        model=models["acdom"]["model"],
        scaler_Rrs=models["acdom"]["scaler_Rrs"],
        TSS_scalers_dict=models["acdom"]["scaler_dict"],
        full_band_wavelengths=SELECTED_BANDS,
        use_diff=False,
        use_spectral_mask=True,
        batch_size=2048,
    )

    products_nc = save_pace_products_to_nc(
        nc_path=nc_path,
        save_dir=save_dir,
        chla_output=chla_output,
        tss_output=tss_output,
        acdom_output=acdom_output,
    )

    acq_date = parse_acquisition_date(nc_path)
    cog_paths = []
    products_ds = xr.open_dataset(products_nc)
    lat_2d = products_ds["latitude"].values
    lon_2d = products_ds["longitude"].values
    for var, label in PRODUCT_LABELS.items():
        cog_paths.append(
            save_product_to_cog(
                out_tif=os.path.join(save_dir, f"PACE_OCI-{acq_date}-{label}.tif"),
                lat_2d=lat_2d,
                lon_2d=lon_2d,
                values_2d=products_ds[var].values,
            )
        )
    products_ds.close()

    # The COGs are the deliverable; drop the intermediate NetCDF to save space.
    if not keep_nc and os.path.exists(products_nc):
        os.remove(products_nc)

    return cog_paths
