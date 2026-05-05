"""
Lädt ein neues Dataset in data/datasets herunter.

Unterstützt:
- Direkte URL (z.B. https://.../dataset.csv)
- Google Drive File ID (ohne zusätzliche Dependencies)
"""

import argparse
import re
from pathlib import Path

import requests


def _download_file(url: str, output_path: Path, session: requests.Session | None = None) -> None:
    sess = session or requests.Session()
    with sess.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _extract_confirm_token(response: requests.Response) -> str | None:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    match = re.search(r"confirm=([0-9A-Za-z_]+)", response.text)
    return match.group(1) if match else None


def download_from_google_drive(file_id: str, output_path: Path) -> None:
    session = requests.Session()
    base_url = "https://drive.google.com/uc?export=download"

    first = session.get(base_url, params={"id": file_id}, timeout=120)
    first.raise_for_status()
    token = _extract_confirm_token(first)

    if token:
        _download_file(f"{base_url}&id={file_id}&confirm={token}", output_path, session=session)
    else:
        # Kleine Dateien benötigen manchmal keinen Confirm-Token.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            file.write(first.content)


def parse_args():
    parser = argparse.ArgumentParser(description="Lädt ein Dataset in den datasets-Ordner.")
    parser.add_argument("--url", help="Direkte Download-URL.")
    parser.add_argument("--gdrive-id", help="Google Drive File-ID.")
    parser.add_argument(
        "--output",
        required=True,
        help="Ausgabedatei (z.B. data/datasets/new_dataset.csv).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)

    if bool(args.url) == bool(args.gdrive_id):
        raise ValueError("Bitte genau eine Option nutzen: --url ODER --gdrive-id")

    if args.url:
        _download_file(args.url, output_path)
    else:
        download_from_google_drive(args.gdrive_id, output_path)

    print(f"[OK] Dataset heruntergeladen: {output_path.resolve()}")


if __name__ == "__main__":
    main()
