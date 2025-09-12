"""
Fetch yearly EAGLE-I eletrical outage CSV files from the Figshare article 

Workflow:
    1. Read a JSON manifest file to get the Figshare article ID and years to download
    2. Call the Figshare public API to list files attached to the article
    3. Match filenames of the form "eaglei_outages_YYYY.csv" and filter them to the years requested
    4. Download each matching file to a local directory structure "data/raw/outages/YYYY
"""
import re
import json
from pathlib import Path

import requests
from tqdm import tqdm

API = "https://api.figshare.com/v2/articles/{id}"

def get_article_files(article_id: int):
    """
    Query the Figshare API for an article's metadata and return its files list
    """
    # Build the article metadata URL 
    url = API.format(id=article_id)

    # Request the JSON metadata
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    # Parse the response body as JSON into a Python dict
    meta = r.json()

    # Extract the files list (default to empty list if missing)
    files = meta.get("files", [])   

    return files


def stream_download(url: str, dest: Path, chunk_bytes: int = 1 << 20):
    """
    Stream a remote file to disk in chucks (default 1MB), creating parent directories
    If the destination file already exists and is non-empty, it is not re-downloaded
    """
    # Ensure the destination directory exists
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Skip redownload if nonempty file is already present
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest)
    
    # Stream the response to avoid loading the entire file into memory
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()

        # If content-length is provided, show a more informative progress bar
        total_bytes = int(r.headers.get("content-length", 0)) if r.headers.get("content-length") else 0
        total_chunks = (total_bytes // chunk_bytes) + (1 if total_bytes % chunk_bytes else 0) if total_bytes else None

        # Write the response content to the destination file in chunks
        with open(dest, "wb") as f, tqdm(
            r.iter_content(chunk_size=chunk_bytes),
            unit="MB",
            total=total_chunks,
            desc=dest.name,
        ) as bar:
            for chunk in bar:
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
    return str(dest)


def load_manifest(path: Path):
    """
    Load the outages manifest JSON
    """
    with open(path, "r") as f:
        return json.load(f)
    
    # Validate expected keys 
    if "figshare_article_id" not in manifest:
        raise KeyError("Manifest missing required key: 'figshare_article_id'")
    if "years" not in manifest:
        raise KeyError("Manifest missing required key: 'years'")

    return manifest


def main():
    """
    Entry point: reads manifest, queries Figshare, downloads matching yearly CSVs
    """
    # Read manifest and extract keys 
    manifest = load_manifest(Path("manifests/outages_manifest.json"))
    article_id = manifest["figshare_article_id"]
    years = set(manifest["years"])

    # Query Figshare API for article files
    files = get_article_files(article_id)

    # Regex to match yearly outage CSV filenames
    pat = re.compile(r"eaglei_outages_(\d{4})\.csv$", re.IGNORECASE)

    found = 0
    for f in files:
        name = f.get("name", "")
        m = pat.search(name)
        if not m:
            # Skip anything that doesn't match the expected filename pattern
            continue
        yr = m.group(1)
        if yr not in years:
            # File year not requested in the manifest, skip it
            continue
        url = f["download_url"]  # public direct link
        dest = Path(f"data/raw/outages/{yr}/{name}")
        print(f"Downloading {name} → {dest}")
        stream_download(url, dest)
        found += 1

    # If nothing matched, alert the user to double-check
    if found == 0:
        raise SystemExit("No yearly CSVs matched—check the Figshare article or regex.")
    print(f"Done. Downloaded {found} files.")

if __name__ == "__main__":
    main()
    