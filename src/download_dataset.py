"""
DemandCast - Dataset Downloader
Fetches the official Indian Electricity Demand, Solar and Wind Generation dataset
from Mendeley Data repository (Identifier: y58jknpgs8 / DOI: 10.17632/y58jknpgs8.2).
"""

import os
import sys
import json
import logging
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DemandCast.Downloader")

MENDELEY_DATASET_ID = "y58jknpgs8"
MENDELEY_API_URL = f"https://data.mendeley.com/public-api/datasets/{MENDELEY_DATASET_ID}"
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


def fetch_dataset_metadata():
    headers = {"User-Agent": "DemandCast-Research-Client/1.0"}
    logger.info("Querying Mendeley Data API for dataset %s...", MENDELEY_DATASET_ID)
    resp = requests.get(MENDELEY_API_URL, headers=headers, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    logger.info("Found dataset: '%s' (Version %s)", data.get("name"), data.get("version"))
    return data


def download_file(url: str, dest_path: str):
    headers = {"User-Agent": "DemandCast-Research-Client/1.0"}
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logger.info("Downloading from %s to %s...", url, dest_path)
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (1024 * 1024 * 5) < 65536:
                        logger.info("Progress: %.1f MB / %.1f MB (%.1f%%)",
                                    downloaded / (1024 * 1024),
                                    total_size / (1024 * 1024),
                                    100.0 * downloaded / total_size)
    logger.info("Download completed successfully: %s (%d bytes)", dest_path, os.path.getsize(dest_path))


def download_mendeley_dataset(target_filename: str = "January 2024- June 2025.xlsx", local_name: str = "Jan2024_June2025.xlsx"):
    dest = os.path.join(RAW_DATA_DIR, local_name)
    if os.path.exists(dest) and os.path.getsize(dest) > 100000:
        logger.info("Target file already exists: %s (%d bytes). Skipping download.", dest, os.path.getsize(dest))
        return dest

    metadata = fetch_dataset_metadata()
    files = metadata.get("files", [])
    target_file = None
    for f in files:
        name = f.get("filename") or f.get("name")
        if name and target_filename.lower() in name.lower():
            target_file = f
            break

    if not target_file:
        # Fallback to the first available valid file or matching content details
        target_file = next((f for f in files if "2024" in str(f.get("filename", ""))), files[0] if files else None)

    if not target_file:
        raise RuntimeError("No suitable files found in Mendeley dataset metadata.")

    download_url = target_file.get("download_url") or (target_file.get("content_details") or {}).get("download_url")
    if not download_url and "id" in target_file:
        download_url = f"https://data.mendeley.com/public-files/datasets/{MENDELEY_DATASET_ID}/files/{target_file['id']}/file_downloaded"

    logger.info("Selected file: %s (ID: %s)", target_file.get("filename"), target_file.get("id"))
    download_file(download_url, dest)
    return dest


if __name__ == "__main__":
    download_mendeley_dataset()
