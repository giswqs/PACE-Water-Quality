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
`output/` folder receives:

- `PACE_OCI.<...>_products.nc` — all three products on the native swath grid
- `PACE_OCI-<YYYYMMDD>-chla.tif`
- `PACE_OCI-<YYYYMMDD>-tss.tif`
- `PACE_OCI-<YYYYMMDD>-acdom.tif`

The date in the COG filenames is parsed from the input filename.

### About the COGs

PACE L2 is swath data with 2D (curvilinear) lat/lon. Each product is
resampled onto a regular EPSG:4326 grid at ~native resolution (~1 km) using
nearest-neighbour interpolation (HyperCoast's `grid_pace` technique), with
cells outside the actual data footprint masked to nodata so values are not
smeared across open water. Each GeoTIFF is written with internal tiling
(512×512), overviews, and DEFLATE compression, then validated with
`rio_cogeo`.

## Automated daily products

A GitHub Actions workflow (`.github/workflows/daily.yml`) runs every day
(and on demand via *Run workflow*). It downloads the most recent PACE scene,
runs inference, and uploads the resulting GeoTIFFs to the repository's
**`latest`** release, so the newest products are always available at:

```
https://github.com/giswqs/PACE-Water-Quality/releases/tag/latest
```

Because output filenames include the acquisition date, products from
different dates accumulate in the release while same-date files are replaced.

### Required repository secrets

The workflow needs NASA Earthdata credentials. Add them under
**Settings → Secrets and variables → Actions**:

- `EARTHDATA_USERNAME`
- `EARTHDATA_PASSWORD`

The `data/` and `output/` folders are git-ignored, so large scenes and
products are never committed; the daily run regenerates them and publishes
only the GeoTIFFs to the release.

## Notes

- Inputs and outputs use paths relative to the scripts, so the project can be
  moved or run from any directory.
- The `model/` weights are required; they are not downloaded automatically.
