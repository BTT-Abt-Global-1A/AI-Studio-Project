"""
Fetch NOAA 'daily-grids/v1-0-0/averages' files for selected years/variables/months/level.

Saves to: data/raw/weather/daily_grids/<year>/
Only downloads files that match: {var}-{YYYY}{MM}-{level}-scaled.csv
Resumable: skips non-empty files.
"""

import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

BASE = "https://www.ncei.noaa.gov/pub/data/daily-grids/v1-0-0/averages/"

def load_manifest(path: Path):
    with open(path, "r") as f:
        m = json.load(f)
    m.setdefault("years", [])
    m.setdefault("variables", [])
    m.setdefault("level", "cty")
    m.setdefault("months", [])        # [] means all months
    m.setdefault("extensions", [".csv"])
    m.setdefault("max_workers", 4)
    return m

def list_year_files(year: str):
    url = f"{BASE}{year}/"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    # capture file links (exclude subdirs ending with "/")
    names = re.findall(r'href="([^"/][^"]+)"', r.text)
    return [(n, url + n) for n in names]

def passes_filters(fname: str, year: str, variables, level: str, months, exts):
    fn = fname.lower()

    # extension filter
    if exts and not any(fn.endswith(ext.lower()) for ext in exts):
        return False

    # variable filter
    if variables and not any(fn.startswith(v.lower() + "-") for v in variables):
        return False

    # year+month filter (YYYYMM)
    # Allow any month if months == []
    m = re.search(r"-(\d{6})-", fn)  # captures YYYYMM between dashes
    if not m:
        return False
    yyyymm = m.group(1)
    if not yyyymm.startswith(str(year)):
        return False
    mm = yyyymm[4:6]
    if months and mm not in months:
        return False

    # level filter (e.g., "-cty-scaled.csv")
    if f"-{level.lower()}-scaled" not in fn:
        return False

    return True

def stream_download(url: str, dest: Path, chunk_bytes: int = 1 << 20):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest)

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total_bytes = int(r.headers.get("content-length", 0)) if r.headers.get("content-length") else 0
        total_chunks = (total_bytes // chunk_bytes) + (1 if total_bytes % chunk_bytes else 0) if total_bytes else None

        with open(dest, "wb") as f, tqdm(
            r.iter_content(chunk_size=chunk_bytes),
            unit="MB",
            total=total_chunks,
            desc=dest.name,
        ) as bar:
            for chunk in bar:
                if chunk:
                    f.write(chunk)
    return str(dest)

def download_year(year: str, variables, level: str, months, exts, max_workers: int):
    entries = list_year_files(year)
    keep = [(name, url) for (name, url) in entries
            if passes_filters(name, year, variables, level, months, exts)]

    if not keep:
        print(f"[{year}] No files matched filters; {len(entries)} files found.")
        return 0

    print(f"[{year}] {len(keep)} files to download (of {len(entries)}).")
    # EXTERNAL DRIVE CONFIGURATION - Update this path for your setup
    # Using external drive "Academia" for large dataset storage
    # Team members: modify this path to match your storage solution
    outdir = Path(f"/Volumes/Academia/AI-Studio-Project/data/raw/weather/daily_grids/{year}")
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for name, url in keep:
            dest = outdir / name
            futures.append(ex.submit(stream_download, url, dest))

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Year {year}"):
            try:
                fut.result()
                done += 1
            except Exception as e:
                print(f"[{year}] Error: {e}")

    return done

def main():
    manifest = load_manifest(Path("manifests/daily_grids_manifest.json"))
    years = [str(y) for y in manifest["years"]]
    variables = [v.lower() for v in manifest["variables"]]
    level = manifest["level"].lower()
    months = manifest["months"]  # e.g., ["08","09","10"] or []
    exts = manifest["extensions"]
    max_workers = int(manifest["max_workers"])

    total = 0
    for y in years:
        total += download_year(y, variables, level, months, exts, max_workers)

    if total == 0:
        raise SystemExit("No files downloaded—check year/variables/level/months filters.")
    print(f"Done. Downloaded {total} files across {len(years)} year(s).")

if __name__ == "__main__":
    main()
