"""Download the latest PACE OCI L2 AOP scene(s) for testing the workflow.

Uses HyperCoast's ``search_pace`` / ``download_pace`` (which wrap NASA
``earthaccess``) to fetch the most recent granule(s) over a region of
interest into the local ``data`` folder. The granules match the same
product type as the bundled test scene (``PACE_OCI.*.L2.OC_AOP.*.nc``), so
they can be fed straight into ``run.py``.

For downloading scenes over a specific date range instead, use
``download_data.py``.

A NASA Earthdata login is required. Credentials are read from ``~/.netrc``
(or the ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD`` environment
variables). Create a free account at https://urs.earthdata.nasa.gov if needed.
"""

import os
import re
from datetime import datetime, timedelta, timezone

import earthaccess
import hypercoast

# Resolve paths relative to this script so it can run from any location.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# === Search parameters ===
# Bounding box over the Gulf of Mexico / U.S. Gulf coast, matching the
# region covered by the bundled test scene. [xmin, ymin, xmax, ymax]
BBOX = (-98.0, 18.0, -80.0, 31.0)
SHORT_NAME = "PACE_OCI_L2_AOP"  # surface reflectance (Rrs) product
VERSION = "V3_2"  # only download this processing version (None keeps all)
NUM_SCENES = 1  # number of most-recent scenes to download
# Look-back windows (days) tried in order until granules are found. PACE
# standard products can lag the present by days/weeks, so widen if needed.
LOOKBACK_DAYS = (7, 30, 90, 365)


def acquisition_time(granule):
    """Extract the acquisition timestamp (YYYYMMDDTHHMMSS) from a granule.

    Args:
        granule (dict): An earthaccess granule result.

    Returns:
        str: The timestamp string, or "" if it cannot be parsed. The string
            sorts chronologically, so it doubles as a sort key.
    """
    native_id = granule.get("meta", {}).get("native-id", "")
    match = re.search(r"(\d{8}T\d{6})", native_id)
    return match.group(1) if match else ""


# === Authenticate with NASA Earthdata ===
# Reads credentials from ~/.netrc or EARTHDATA_USERNAME/PASSWORD env vars.
earthaccess.login(persist=True)

# === Search recent windows until granules are found ===
end = datetime.now(timezone.utc)
granules = []
for days in LOOKBACK_DAYS:
    start = end - timedelta(days=days)
    temporal = (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    print(f"Searching {temporal[0]} to {temporal[1]} ...")
    granules = hypercoast.search_pace(
        bbox=BBOX,
        temporal=temporal,
        count=-1,
        short_name=SHORT_NAME,
    )
    # Keep only the requested processing version (e.g. V3_2).
    if VERSION is not None:
        granules = [
            g
            for g in granules
            if f".{VERSION}." in g.get("meta", {}).get("native-id", "")
        ]
    if granules:
        break

if not granules:
    raise RuntimeError(
        f"No PACE {VERSION or ''} granules found over the bounding box in the "
        f"last {LOOKBACK_DAYS[-1]} days."
    )

# Keep the newest version per acquisition time, then take the latest scenes.
by_time = {}
for g in granules:
    t = acquisition_time(g)
    native_id = g.get("meta", {}).get("native-id", "")
    if t and (t not in by_time or native_id > by_time[t].get("meta", {}).get("native-id", "")):
        by_time[t] = g

latest = [by_time[t] for t in sorted(by_time, reverse=True)][:NUM_SCENES]

print(f"\nLatest {len(latest)} scene(s):")
for g in latest:
    print("  ", g["meta"]["native-id"])

# === Download ===
files = hypercoast.download_pace(latest, out_dir=DATA_DIR)
print(f"\nDownloaded {len(files)} file(s) to {DATA_DIR}:")
for f in files:
    print("  ", os.path.basename(f))
