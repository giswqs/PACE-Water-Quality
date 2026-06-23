# PACE Water-Quality Products (MoE-VAE)

Generate gridded water-quality products from NASA **PACE OCI L2 AOP**
(surface reflectance, `Rrs`) scenes using a Mixture-of-Experts Variational
Autoencoder (MoE-VAE). For each scene the workflow estimates three products
and writes both a multi-variable NetCDF and validated Cloud Optimized
GeoTIFFs (COGs):

| Product   | Variable    | Units      |
|-----------|-------------|------------|
| Chlorophyll-a | `chla`      | mg m⁻³ |
| Total Suspended Solids | `tss`       | g m⁻³  |
| CDOM absorption @ 440 nm | `acdom440`  | m⁻¹    |

## Project layout

```
PACE_PRODUCT/
├── code/                  # MoE-VAE model + PACE inference/IO helpers
├── model/                 # Trained weights & scalers (chl-a / tss / acdom)
├── data/                  # Input PACE L2 AOP NetCDF scenes
├── output/                # Generated products (NetCDF + COGs)
├── download_data.py       # Download scenes over a specified date range
├── download_latest.py     # Download the most recent scene
├── pace_processing.py     # Shared logic: load_models / process_scene / COG
├── run_file.py            # Process a single scene
├── run_folder.py          # Process every scene in a folder
├── run.py                 # Original single-scene script (self-contained)
├── requirements.txt
└── README.md
```

## Installation

Python 3.10+ with a CUDA-capable GPU recommended (CPU works but is slower).

```bash
pip install -r requirements.txt
```

## NASA Earthdata credentials

Downloading requires a free [Earthdata](https://urs.earthdata.nasa.gov)
account. Store credentials in `~/.netrc`:

```
machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD
```

(or set the `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` environment variables).

## Usage

### 1. Download data

```bash
# Latest available scene over the region of interest
python download_latest.py

# Scenes over a specific date range (Gulf of Mexico by default)
python download_data.py 2024-07-01 2024-07-31
python download_data.py 2024-09-01 2024-09-30 --count 5
python download_data.py 2024-09-29 2024-09-29 --bbox -99 18 -78 42 --version V3_2
```

Both scripts save into `data/` and, by default, fetch only the **V3_2**
processing version. `download_data.py` options: `--count`, `--bbox`,
`--short-name`, `--version` (use `--version all` to keep every version).

### 2. Process scenes

```bash
# A single scene (path, or a filename found in data/)
python run_file.py PACE_OCI.20240929T185124.L2.OC_AOP.V3_0.nc
python run_file.py data/PACE_OCI.20240701T175112.L2.OC_AOP.V3_1.nc

# Every scene in a folder (defaults to data/ -> output/)
python run_folder.py
python run_folder.py data --output results --pattern "PACE_OCI.*V3_2.nc"
```

Both accept `--output` and `--model-dir`. `run_folder.py` loads the models
once, skips any `*_products.nc` files, and continues past individual scene
failures (reporting them in a summary).

## Outputs

For an input named `PACE_OCI.<YYYYMMDD>T<HHMMSS>.L2.OC_AOP.<ver>.nc`, the
`output/` folder receives one date-named COG per product:

- `PACE_OCI-<YYYYMMDD>-chla.tif`
- `PACE_OCI-<YYYYMMDD>-tss.tif`
- `PACE_OCI-<YYYYMMDD>-acdom.tif`

The date is parsed from the input filename. When several PACE passes share a
date, the pass with the most valid retrieval pixels is kept (best pass per
day).

### About the COGs

PACE L2 is swath data with 2D (curvilinear) lat/lon. Each product is written
**directly** from the swath array to a GeoTIFF whose transform spans the
swath's lon/lat bounds (`rasterio.transform.from_bounds`) — every pixel keeps
its exact model value, with **no gridding, gap-filling or interpolation**, so
no synthetic values are introduced (invalid pixels are nodata). Each GeoTIFF
is written with internal tiling, overviews and DEFLATE compression, then
validated with `rio_cogeo`.

Inference is deterministic: the models run in `eval()` mode, which disables
the MoE noisy gating and makes the VAE use its latent mean, so re-running a
scene reproduces the same products.

## Automated daily products

A GitHub Actions workflow (`.github/workflows/daily.yml`) runs every day
(and on demand via *Run workflow*). It downloads the most recent PACE scene,
runs inference, and publishes the resulting GeoTIFFs to two places:

- the repository's **`PACE-Data`** release:
  https://github.com/giswqs/PACE-Water-Quality/releases/tag/PACE-Data
- the **Hugging Face dataset** (under `cogs/`):
  https://huggingface.co/datasets/giswqs/PACE-Water-Quality

Because output filenames include the acquisition date, products from
different dates accumulate while same-date files are replaced.

### Required repository secrets

The workflow needs the following secrets under
**Settings → Secrets and variables → Actions**:

- `EARTHDATA_USERNAME` — NASA Earthdata login
- `EARTHDATA_PASSWORD` — NASA Earthdata password
- `HF_TOKEN` — Hugging Face token with write access to the dataset

The `data/` and `output/` folders are git-ignored, so large scenes and
products are never committed; the daily run regenerates them and publishes
only the GeoTIFFs to the release.

## Notes

- Inputs and outputs use paths relative to the scripts, so the project can be
  moved or run from any directory.
- The `model/` weights are required; they are not downloaded automatically.
