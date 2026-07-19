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
├── moe_vae/               # MoE-VAE model + PACE inference/IO helpers
├── model/                 # Trained weights & scalers (chl-a / tss / acdom)
├── data/                  # Input PACE L2 AOP NetCDF scenes
├── output/                # Generated products (NetCDF + COGs)
├── json/                  # Per-date GeoJSON catalogs of the published COGs
├── download_data.py       # Download scenes over a specified date range
├── download_latest.py     # Download every pass from the most recent date
├── pace_processing.py     # Shared logic: load_models / process_scene / COG
├── run_file.py            # Process a single scene
├── run_folder.py          # Process every scene in a folder
├── make_json.py           # Build the per-date JSON catalogs from the COGs
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
# Every pass from the most recent available date over the region of interest
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

# Process and build the per-date JSON catalogs in one go
python run_folder.py --json-dir json
```

Both accept `--output` and `--model-dir`. `run_folder.py` loads the models
once, skips any `*_products.nc` files, and continues past individual scene
failures (reporting them in a summary). Scenes whose COGs already exist are
skipped unless `--overwrite` is passed, so an interrupted backfill can just
be re-run. `--limit N` processes only the first N pending scenes, which is
handy for a quick test.

### 3. Build the JSON catalogs

```bash
python make_json.py                                  # ./output -> ./json
python make_json.py /path/to/output --json-dir /path/to/json
python make_json.py --base-url https://example.com/data
```

`make_json.py` reads the COGs on disk, so it can be re-run at any time
without reprocessing.

## Outputs

Each input granule produces one COG per product, named after the granule and
placed in a per-product subfolder. For an input
`PACE_OCI.20240929T185124.L2.OC_AOP.V3_2.nc`:

```
output/chla/PACE_OCI.20240929T185124.L2.OC_AOP.V3_2.tif
output/tss/PACE_OCI.20240929T185124.L2.OC_AOP.V3_2.tif
output/acdom/PACE_OCI.20240929T185124.L2.OC_AOP.V3_2.tif
```

Every pass is kept. PACE often crosses the region several times a day, so a
date with four passes yields four COGs per product — nothing is collapsed to
a single "best pass".

### JSON catalogs

`make_json.py` writes one catalog per acquisition date and product,
`json/<YYYYMMDD>_<product>.json`, listing every pass from that date as a
minimal GeoJSON `FeatureCollection` (the same shape as
[this NAIP catalog](https://data.source.coop/giswqs/opengeos/naip_nd_2023_stac.json)):

```json
{"type":"FeatureCollection","features":[{"bbox":[-96.769775,-1.638628,-67.06871,26.577588],"assets":{"image":{"href":"https://huggingface.co/datasets/giswqs/PACE-Water-Quality/resolve/main/data/chla/PACE_OCI.20260322T181227.L2.OC_AOP.V3_2.tif"}}}]}
```

The files are written compact (no indentation), like the reference catalog.

The `bbox` is read from the COG and the `href` points at the published copy.
The base URL defaults to the Hugging Face dataset and can be changed with
`--base-url`.

### About the COGs

PACE L2 swaths are rotated/curved in lon/lat, so the array's `(row, col)`
layout is not axis-aligned and cannot be written to a GeoTIFF directly
(`from_bounds` on the raw array mis-georeferences it). Instead each product is
gridded at its true `(lon, lat)` onto a regular EPSG:4326 grid (~1 km, 0.01°)
with `scipy.interpolate.griddata` — the upstream `npy_to_tif` approach. This
georeferences correctly and the interpolation fills the thin rotated-scan
gaps for a continuous coastal field, while leaving the open ocean / large
cloud gaps outside the data hull as nodata. Each GeoTIFF is written with
internal tiling, overviews and DEFLATE compression, then validated with
`rio_cogeo`.

Inference is deterministic: the models run in `eval()` mode, which disables
the MoE noisy gating and makes the VAE use its latent mean, so re-running a
scene reproduces the same products.

## Automated daily products

A GitHub Actions workflow (`.github/workflows/daily.yml`) runs every day
(and on demand via *Run workflow*). It downloads every PACE pass from the
most recent available date, runs inference, and publishes the results to the
**Hugging Face dataset** https://huggingface.co/datasets/giswqs/PACE-Water-Quality:

- COGs under `data/<product>/`:
  https://huggingface.co/datasets/giswqs/PACE-Water-Quality/resolve/main/data/
- JSON catalogs under `json/`:
  https://huggingface.co/datasets/giswqs/PACE-Water-Quality/resolve/main/json/

Because COG filenames are the granule names, products accumulate without
ever colliding, and re-running a scene simply replaces its own files.

### Required repository secrets

The workflow needs the following secrets under
**Settings → Secrets and variables → Actions**:

- `EARTHDATA_USERNAME` — NASA Earthdata login
- `EARTHDATA_PASSWORD` — NASA Earthdata password
- `HF_TOKEN` — Hugging Face token with write access to the dataset

The `data/`, `output/` and `json/` folders are git-ignored, so large scenes
and products are never committed; the daily run regenerates them and
publishes the COGs and catalogs to Hugging Face.

## Notes

- Inputs and outputs use paths relative to the scripts, so the project can be
  moved or run from any directory.
- The `model/` weights are required; they are not downloaded automatically.
