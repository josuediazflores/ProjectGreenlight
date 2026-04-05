"""
Scrape EB-1A (Extraordinary Ability) AAO Non-Precedent Decisions from USCIS.

Downloads all PDF decisions from the USCIS AAO repository filtered to
I-140 Extraordinary Ability (topic filter value 19).

Usage:
    python scripts/scrape_aao.py
    python scripts/scrape_aao.py --items-per-page 50 --delay 2.0
"""

import argparse
import csv
import os
import time
from pathlib import Path
from urllib.parse import urljoin, unquote

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://www.uscis.gov/administrative-appeals/aao-decisions/aao-non-precedent-decisions"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_pdfs"
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data" / "manifest.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) immigration-ai-research"
}


def get_page_url(page: int, items_per_page: int = 100) -> str:
    """Build the URL for a given page of EB-1A results."""
    params = f"?uri_1=19&m=All&y=All&items_per_page={items_per_page}&page={page}"
    return BASE_URL + params


def parse_pdf_links(html: str) -> list[dict]:
    """Extract PDF links and metadata from a results page."""
    soup = BeautifulSoup(html, "html.parser")
    entries = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if not href.endswith(".pdf"):
            continue

        # Build absolute URL
        url = urljoin("https://www.uscis.gov", href)
        filename = unquote(url.split("/")[-1])
        title = link.get_text(strip=True)

        # Try to find the date near this link
        parent = link.find_parent("td") or link.find_parent("div")
        date_text = ""
        if parent:
            text = parent.get_text(" ", strip=True)
            # Dates appear as "Month DD, YYYY"
            import re
            date_match = re.search(
                r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
                text,
            )
            if date_match:
                date_text = date_match.group(0)

        entries.append({
            "filename": filename,
            "url": url,
            "date": date_text,
            "title": title,
        })

    return entries


def load_existing_manifest() -> set[str]:
    """Load already-downloaded filenames from manifest."""
    if not MANIFEST_PATH.exists():
        return set()
    with open(MANIFEST_PATH) as f:
        reader = csv.DictReader(f)
        return {row["filename"] for row in reader}


def download_pdf(url: str, filepath: Path, session: requests.Session) -> bool:
    """Download a single PDF. Returns True on success."""
    try:
        resp = session.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        filepath.write_bytes(resp.content)
        return True
    except requests.RequestException as e:
        print(f"  Failed to download {url}: {e}")
        return False


def scrape(items_per_page: int = 100, delay: float = 1.5, max_pages: int | None = None):
    """Main scraping loop."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    existing = load_existing_manifest()
    print(f"Found {len(existing)} already-downloaded PDFs in manifest.")

    session = requests.Session()

    # Initialize manifest CSV
    write_header = not MANIFEST_PATH.exists()
    manifest_file = open(MANIFEST_PATH, "a", newline="")
    writer = csv.DictWriter(manifest_file, fieldnames=["filename", "url", "date", "title"])
    if write_header:
        writer.writeheader()

    page = 0
    total_downloaded = 0
    total_skipped = 0
    consecutive_empty = 0

    try:
        while True:
            if max_pages is not None and page >= max_pages:
                print(f"Reached max pages limit ({max_pages}).")
                break

            url = get_page_url(page, items_per_page)
            print(f"\nFetching page {page}: {url}")

            try:
                resp = session.get(url, headers=HEADERS, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as e:
                print(f"  Failed to fetch page {page}: {e}")
                break

            entries = parse_pdf_links(resp.text)

            if not entries:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    print("No more results. Done.")
                    break
                page += 1
                continue

            consecutive_empty = 0
            print(f"  Found {len(entries)} PDFs on this page.")

            for entry in tqdm(entries, desc=f"Page {page}", leave=False):
                if entry["filename"] in existing:
                    total_skipped += 1
                    continue

                filepath = DATA_DIR / entry["filename"]
                if download_pdf(entry["url"], filepath, session):
                    writer.writerow(entry)
                    manifest_file.flush()
                    existing.add(entry["filename"])
                    total_downloaded += 1

                time.sleep(delay)

            page += 1
            time.sleep(delay)

    finally:
        manifest_file.close()

    print(f"\nDone! Downloaded: {total_downloaded}, Skipped (existing): {total_skipped}")
    print(f"Total PDFs in manifest: {len(existing)}")


def main():
    parser = argparse.ArgumentParser(description="Scrape EB-1A AAO decisions from USCIS")
    parser.add_argument("--items-per-page", type=int, default=100, help="Results per page (default: 100)")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between requests in seconds (default: 1.5)")
    parser.add_argument("--max-pages", type=int, default=None, help="Max pages to scrape (default: all)")
    args = parser.parse_args()

    scrape(items_per_page=args.items_per_page, delay=args.delay, max_pages=args.max_pages)


if __name__ == "__main__":
    main()
