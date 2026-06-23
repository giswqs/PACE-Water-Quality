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
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
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

    # eval() mode: disables noisy gating and makes the VAE use the latent
    # mean (deterministic inference).
    for mdl in (chla_model, tss_model, acdom_model):
        mdl.eval()

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


def save_product_to_cog(out_tif, lat_2d, lon_2d, values_2d, nodata=-9999.0):
    """Write a PACE swath product to a COG directly, without interpolation.

    This adopts the upstream ``save_pace_products_to_tif`` approach: the 2D
    swath array is written straight to a GeoTIFF whose transform spans the
    swath's lon/lat bounds (``rasterio.transform.from_bounds``). Each pixel
    keeps its exact model value -- there is no gridding, gap-filling or
    interpolation, so no synthetic values are introduced; invalid pixels
    become nodata. The array is flipped if needed so north is up, then
    written as a Cloud Optimized GeoTIFF (internal tiling, overviews, DEFLATE
    compression) and validated.

    Args:
        out_tif (str): Output GeoTIFF path.
        lat_2d (np.ndarray): 2D latitude array (degrees north, EPSG:4326).
        lon_2d (np.ndarray): 2D longitude array (degrees east, EPSG:4326).
        values_2d (np.ndarray): 2D product values aligned with lat/lon.
        nodata (float): Value used for invalid/empty pixels.

    Returns:
        str: The path to the validated COG.
    """
    lat = np.asarray(lat_2d, dtype=np.float64)
    lon = np.asarray(lon_2d, dtype=np.float64)
    data = np.asarray(values_2d, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("values_2d must be a 2D swath array")

    # Make sure north is up (PACE swaths may be stored south-first).
    if np.nanmean(lat[0, :]) < np.nanmean(lat[-1, :]):
        data = data[::-1, :]
        lat = lat[::-1, :]
        lon = lon[::-1, :]

    height, width = data.shape
    lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
    lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)

    valid = int(np.isfinite(data).sum())
    data = np.where(np.isfinite(data), data, nodata).astype(np.float32)

    src_profile = dict(
        driver="GTiff",
        dtype="float32",
        count=1,
        height=height,
        width=width,
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
    )
    dst_profile = cog_profiles.get("deflate")
    with MemoryFile() as mem:
        with mem.open(**src_profile) as src:
            src.write(data, 1)
        with mem.open() as src:
            cog_translate(
                src,
                out_tif,
                dst_profile,
                overview_resampling="nearest",
                quiet=True,
            )

    is_valid, errors, warnings = cog_validate(out_tif)
    status = "valid" if is_valid else "INVALID"
    print(f"COG {status}: {out_tif} ({valid} valid pixels)")
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


def infer_scene_maps(nc_path, models):
    """Run inference on one PACE scene and return the product maps in memory.

    No files are written. The per-pixel model outputs are reshaped to the
    scene's native swath grid so they can be written directly to GeoTIFFs
    (no gridding/interpolation).

    Args:
        nc_path (str): Path to the input PACE L2 AOP NetCDF file.
        models (dict): Loaded models/scalers from :func:`load_models`.

    Returns:
        dict: ``{"latitude", "longitude", "chla", "tss", "acdom440",
            "valid"}`` where the first five are 2D arrays and ``valid`` is the
            number of valid (finite) retrieval pixels.
    """
    import hypercoast

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

    da = hypercoast.read_pace(nc_path)["Rrs"]
    lat = da.latitude.values
    lon = da.longitude.values
    shape = lat.shape

    chla = chla_output[:, 2].reshape(shape).astype(np.float32)
    tss = tss_output[:, 2].reshape(shape).astype(np.float32)
    acdom = acdom_output[:, 2].reshape(shape).astype(np.float32)

    return {
        "latitude": lat,
        "longitude": lon,
        "chla": chla,
        "tss": tss,
        "acdom440": acdom,
        "valid": int(np.isfinite(chla).sum()),
    }


def write_scene_cogs(maps, save_dir, date):
    """Write the in-memory product maps to date-named direct COGs.

    Args:
        maps (dict): Output of :func:`infer_scene_maps`.
        save_dir (str): Output directory.
        date (str): Acquisition date (YYYYMMDD) used in the filename.

    Returns:
        list[str]: Paths to the written COGs.
    """
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    for var, label in PRODUCT_LABELS.items():
        paths.append(
            save_product_to_cog(
                out_tif=os.path.join(save_dir, f"PACE_OCI-{date}-{label}.tif"),
                lat_2d=maps["latitude"],
                lon_2d=maps["longitude"],
                values_2d=maps[var],
            )
        )
    return paths


def process_scene(nc_path, models, save_dir):
    """Run inference on one PACE scene and write direct (no-interp) COGs.

    Args:
        nc_path (str): Path to the input PACE L2 AOP NetCDF file.
        models (dict): Loaded models/scalers from :func:`load_models`.
        save_dir (str): Directory to write the products into.

    Returns:
        list[str]: Paths to the written COG files.
    """
    print(f"Processing scene: {nc_path}")
    maps = infer_scene_maps(nc_path, models)
    return write_scene_cogs(maps, save_dir, parse_acquisition_date(nc_path))
