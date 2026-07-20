"""Backfill the full PACE archive one month at a time.

Downloading the entire mission at once would need ~500 GB of scratch space and
publishes nothing until the very end. This driver instead walks the archive
month by month, and for each month:

1. downloads that month's granules,
2. runs inference (scenes that already have COGs are skipped),
3. verifies every granule produced three valid COGs,
4. rebuilds the catalogs for the affected dates,
5. uploads the month's COGs and catalogs to Hugging Face,
6. deletes the month's NetCDF granules, but only the ones whose COGs were
   verified in step 3.

Peak disk use therefore stays at roughly one month of granules (~25 GB) and
results become available as they land rather than days later.

Progress is recorded in a state file, so an interrupted backfill resumes at
the first unfinished month. Re-running a completed month is a no-op.

Examples::

    # Whole archive, from the first PACE scene to the latest available
    python backfill.py --start 2024-03 --end 2026-06

    # A few months, without touching Hugging Face
    python backfill.py --start 2024-03 --end 2024-05 --no-upload

    # See what would happen, without downloading or deleting anything
    python backfill.py --start 2024-03 --end 2026-06 --dry-run

A NASA Earthdata login (``~/.netrc``) and, unless ``--no-upload`` is given, a
Hugging Face token are required.
"""

import os
import re
import sys
import glob
import json
import argparse
import subprocess
from datetime import date, timedelta

from pace_processing import BASE_DIR, PRODUCT_LABELS, HF_DATA_URL, scene_stem

HF_REPO = "giswqs/PACE-Water-Quality"
DEFAULT_BBOX = (-98.0, 18.0, -80.0, 31.0)


def month_range(start, end):
    """List the months spanned by two ``YYYY-MM`` strings, inclusive.

    Args:
        start (str): First month, ``"YYYY-MM"``.
        end (str): Last month, ``"YYYY-MM"``.

    Returns:
        list[tuple[int, int]]: ``(year, month)`` pairs in chronological order.
    """
    sy, sm = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def month_bounds(year, month):
    """Return the first and last calendar day of a month.

    Args:
        year (int): Four-digit year.
        month (int): Month number (1-12).

    Returns:
        tuple[str, str]: ``(first_day, last_day)`` as ``YYYY-MM-DD``.
    """
    first = date(year, month, 1)
    last = date(year + (month == 12), (month % 12) + 1, 1) - timedelta(days=1)
    return first.isoformat(), last.isoformat()


def load_state(path):
    """Read the set of months already completed.

    Args:
        path (str): Path to the JSON state file.

    Returns:
        set[str]: Completed months as ``"YYYY-MM"`` strings.
    """
    if not os.path.isfile(path):
        return set()
    with open(path) as f:
        return set(json.load(f).get("done", []))


def save_state(path, done):
    """Persist the set of completed months.

    Args:
        path (str): Path to the JSON state file.
        done (set[str]): Completed months.
    """
    with open(path, "w") as f:
        json.dump({"done": sorted(done)}, f, indent=1)


def run(cmd, dry_run=False):
    """Run a subprocess, raising if it fails.

    Args:
        cmd (list[str]): Command and arguments.
        dry_run (bool): If True, only print the command.

    Raises:
        subprocess.CalledProcessError: If the command exits non-zero.
    """
    print("  $", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def month_granules(data_dir, year, month):
    """List the downloaded granules belonging to one month.

    Args:
        data_dir (str): Directory holding the NetCDF granules.
        year (int): Four-digit year.
        month (int): Month number.

    Returns:
        list[str]: Paths to the month's ``.nc`` files.
    """
    prefix = f"{year:04d}{month:02d}"
    return sorted(
        p
        for p in glob.glob(os.path.join(data_dir, "*.nc"))
        if re.search(rf"\.{prefix}\d{{2}}T\d{{6}}\.", os.path.basename(p))
    )


def verify_cogs(granules, output_dir):
    """Check that every granule produced three valid COGs.

    Args:
        granules (list[str]): Paths to the month's NetCDF granules.
        output_dir (str): Root directory holding ``<product>/*.tif``.

    Returns:
        tuple[list[str], list[str]]: ``(verified, missing)`` granule paths.
            ``verified`` granules are safe to delete.
    """
    from rio_cogeo.cogeo import cog_validate

    verified, missing = [], []
    for nc in granules:
        stem = scene_stem(nc)
        ok = True
        for label in sorted(set(PRODUCT_LABELS.values())):
            tif = os.path.join(output_dir, label, f"{stem}.tif")
            if not os.path.isfile(tif) or not cog_validate(tif, quiet=True)[0]:
                ok = False
                break
        (verified if ok else missing).append(nc)
    return verified, missing


def upload_month(output_dir, json_dir, granules, dry_run=False):
    """Upload a month's COGs and catalogs to the Hugging Face dataset.

    Args:
        output_dir (str): Root directory holding ``<product>/*.tif``.
        json_dir (str): Directory holding the ``<date>_<product>.json`` files.
        granules (list[str]): The month's granules, used to select which COGs
            and catalogs belong to this month.
        dry_run (bool): If True, only report what would be uploaded.
    """
    from huggingface_hub import HfApi

    labels = sorted(set(PRODUCT_LABELS.values()))
    stems = [scene_stem(nc) for nc in granules]
    cogs = [f"{lab}/{s}.tif" for s in stems for lab in labels]
    dates = sorted({re.search(r"(\d{8})T", s).group(1) for s in stems})
    cats = [f"{d}_{lab}.json" for d in dates for lab in labels]

    print(f"  uploading {len(cogs)} COGs and {len(cats)} catalogs", flush=True)
    if dry_run:
        return
    api = HfApi()
    api.upload_folder(
        folder_path=output_dir,
        path_in_repo="data",
        repo_id=HF_REPO,
        repo_type="dataset",
        allow_patterns=cogs,
        commit_message=f"Add PACE COGs for {dates[0][:6]}",
    )
    api.upload_folder(
        folder_path=json_dir,
        path_in_repo="json",
        repo_id=HF_REPO,
        repo_type="dataset",
        allow_patterns=cats,
        commit_message=f"Add PACE catalogs for {dates[0][:6]}",
    )


def main():
    """Run the month-by-month backfill."""
    parser = argparse.ArgumentParser(
        description="Backfill the PACE archive one month at a time."
    )
    parser.add_argument("--start", required=True, help="First month (YYYY-MM).")
    parser.add_argument("--end", required=True, help="Last month (YYYY-MM).")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(BASE_DIR, "data"),
        help="Directory for the downloaded granules (default: ./data).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(BASE_DIR, "output"),
        help="Root directory for the COGs (default: ./output).",
    )
    parser.add_argument(
        "--json-dir",
        default=os.path.join(BASE_DIR, "json"),
        help="Directory for the catalogs (default: ./json).",
    )
    parser.add_argument(
        "--model-dir",
        default=os.path.join(BASE_DIR, "model"),
        help="Directory containing the model subfolders (default: ./model).",
    )
    parser.add_argument(
        "--state",
        default=None,
        help="Progress file (default: <data-dir>/../backfill_state.json).",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        default=DEFAULT_BBOX,
        help="Bounding box (default: Gulf of Mexico).",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip the Hugging Face upload step.",
    )
    parser.add_argument(
        "--keep-nc",
        action="store_true",
        help="Keep the NetCDF granules instead of deleting them once their "
        "COGs are verified.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what each month would do without downloading, "
        "processing, uploading or deleting.",
    )
    args = parser.parse_args()

    state_path = args.state or os.path.join(
        os.path.dirname(os.path.abspath(args.data_dir)), "backfill_state.json"
    )
    done = load_state(state_path)
    months = month_range(args.start, args.end)
    print(f"Backfill {args.start} .. {args.end}: {len(months)} month(s)")
    print(f"State file: {state_path} ({len(done)} month(s) already done)\n")

    for year, month in months:
        tag = f"{year:04d}-{month:02d}"
        if tag in done:
            print(f"[{tag}] already done, skipping")
            continue
        first, last = month_bounds(year, month)
        print(f"\n=== [{tag}] {first} .. {last} ===", flush=True)

        # 1. Download this month's granules.
        run(
            [
                sys.executable,
                os.path.join(BASE_DIR, "download_data.py"),
                first,
                last,
                "--count",
                "-1",
                "--bbox",
                *[str(v) for v in args.bbox],
                "--out-dir",
                args.data_dir,
            ],
            args.dry_run,
        )

        granules = month_granules(args.data_dir, year, month)
        print(f"  {len(granules)} granule(s) on disk for {tag}")
        if not granules and not args.dry_run:
            print(f"  no granules for {tag}; marking done")
            done.add(tag)
            save_state(state_path, done)
            continue

        # 2. Inference (already-processed scenes are skipped internally).
        run(
            [
                sys.executable,
                "-u",
                os.path.join(BASE_DIR, "run_folder.py"),
                args.data_dir,
                "--output",
                args.output,
                "--model-dir",
                args.model_dir,
            ],
            args.dry_run,
        )

        # 3. Verify before anything irreversible happens.
        if args.dry_run:
            verified, missing = granules, []
        else:
            verified, missing = verify_cogs(granules, args.output)
        print(f"  verified {len(verified)}/{len(granules)} granule(s)")
        for nc in missing:
            print(f"    INCOMPLETE: {os.path.basename(nc)}")

        # 4. Rebuild the catalogs (cheap, and keeps them consistent).
        run(
            [
                sys.executable,
                os.path.join(BASE_DIR, "make_json.py"),
                args.output,
                "--json-dir",
                args.json_dir,
                "--base-url",
                HF_DATA_URL,
            ],
            args.dry_run,
        )

        # 5. Publish.
        if not args.no_upload and verified:
            upload_month(args.output, args.json_dir, verified, args.dry_run)

        # 6. Reclaim disk, but only for granules whose COGs are verified.
        if not args.keep_nc:
            print(f"  deleting {len(verified)} verified granule(s)")
            if not args.dry_run:
                for nc in verified:
                    os.remove(nc)
        if missing:
            print(f"  keeping {len(missing)} granule(s) with incomplete COGs")

        # 7. A month counts as done only if every granule was verified.
        if not missing:
            done.add(tag)
            if not args.dry_run:
                save_state(state_path, done)
        else:
            print(f"  [{tag}] NOT marked done ({len(missing)} incomplete)")

    print(f"\nBackfill finished. {len(done)}/{len(months)} month(s) complete.")


if __name__ == "__main__":
    main()
