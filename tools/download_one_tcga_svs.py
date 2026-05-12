import json
import os
from pathlib import Path

import requests


GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"

OUT_DIR = Path("data/raw/tcga/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_SIZE = 800_000_000       # 800 MB
MAX_SIZE = 1_200_000_000     # 1.2 GB

PROJECTS = [
    "TCGA-BRCA",
    "TCGA-LUAD",
    "TCGA-COAD",
    "TCGA-PRAD",
]


filters = {
    "op": "and",
    "content": [
        {
            "op": "in",
            "content": {
                "field": "files.data_type",
                "value": ["Slide Image"],
            },
        },
        {
            "op": "in",
            "content": {
                "field": "files.data_format",
                "value": ["SVS"],
            },
        },
        {
            "op": "in",
            "content": {
                "field": "files.access",
                "value": ["open"],
            },
        },
        {
            "op": "in",
            "content": {
                "field": "cases.project.project_id",
                "value": PROJECTS,
            },
        },
        {
            "op": ">=",
            "content": {
                "field": "files.file_size",
                "value": MIN_SIZE,
            },
        },
        {
            "op": "<=",
            "content": {
                "field": "files.file_size",
                "value": MAX_SIZE,
            },
        },
    ],
}

fields = [
    "file_id",
    "file_name",
    "file_size",
    "data_type",
    "data_format",
    "access",
    "cases.project.project_id",
    "cases.submitter_id",
]

params = {
    "filters": json.dumps(filters),
    "fields": ",".join(fields),
    "format": "JSON",
    "size": "10",
    "sort": "file_size:asc",
}

print("Searching GDC for open-access SVS files near 1GB...")

response = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=60)
response.raise_for_status()

hits = response.json()["data"]["hits"]

if not hits:
    raise SystemExit("No matching SVS files found. Try widening MIN_SIZE/MAX_SIZE.")

print("\nCandidates:")
for i, item in enumerate(hits):
    size_gb = item["file_size"] / 1_000_000_000
    project = item.get("cases", [{}])[0].get("project", {}).get("project_id", "unknown")
    print(f"{i}: {item['file_name']} | {size_gb:.2f} GB | {project} | {item['file_id']}")

selected = hits[0]

file_id = selected["file_id"]
file_name = selected["file_name"]
out_path = OUT_DIR / file_name

print(f"\nDownloading:")
print(f"  {file_name}")
print(f"  {selected['file_size'] / 1_000_000_000:.2f} GB")
print(f"  to {out_path}")

download_url = f"{GDC_DATA_ENDPOINT}/{file_id}"

with requests.get(download_url, stream=True, timeout=120) as r:
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    downloaded = 0

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue

            f.write(chunk)
            downloaded += len(chunk)

            if total:
                pct = downloaded / total * 100
                print(f"\rDownloaded {downloaded / 1_000_000_000:.2f} GB / {total / 1_000_000_000:.2f} GB ({pct:.1f}%)", end="")
            else:
                print(f"\rDownloaded {downloaded / 1_000_000_000:.2f} GB", end="")

print("\n\nDone.")
print(f"Saved to: {out_path}")
