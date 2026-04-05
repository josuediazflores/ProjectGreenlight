"""
Extract text from AAO decision PDFs.

Uses PyMuPDF (fitz) as primary extractor. Falls back to Tesseract OCR
for scanned/image-based PDFs that yield little or no text.

Usage:
    python scripts/parse_pdfs.py
    python scripts/parse_pdfs.py --ocr-threshold 100  # min chars before OCR fallback
"""

import argparse
import csv
import json
from pathlib import Path

import fitz  # pymupdf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw_pdfs"
TEXT_DIR = DATA_DIR / "extracted_text"
PARSE_META_PATH = DATA_DIR / "parse_metadata.csv"


def extract_text_pymupdf(pdf_path: Path) -> str:
    """Extract text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


def extract_text_ocr(pdf_path: Path) -> str:
    """Extract text from a scanned PDF using Tesseract OCR via PyMuPDF."""
    try:
        import pytesseract
        from PIL import Image
        import io
    except ImportError:
        print(f"  OCR dependencies not available, skipping {pdf_path.name}")
        return ""

    doc = fitz.open(pdf_path)
    text_parts = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render page to image at 300 DPI
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img)
        text_parts.append(text)

    doc.close()
    return "\n".join(text_parts)


def parse_all(ocr_threshold: int = 100):
    """Parse all PDFs in raw_pdfs/ and output text files."""
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in data/raw_pdfs/. Run scrape_aao.py first.")
        return

    # Load already-parsed files
    existing = set()
    if PARSE_META_PATH.exists():
        with open(PARSE_META_PATH) as f:
            reader = csv.DictReader(f)
            existing = {row["filename"] for row in reader}

    to_parse = [p for p in pdfs if p.stem not in existing]
    print(f"Found {len(pdfs)} PDFs total, {len(to_parse)} to parse.")

    write_header = not PARSE_META_PATH.exists()
    meta_file = open(PARSE_META_PATH, "a", newline="")
    writer = csv.DictWriter(meta_file, fieldnames=["filename", "page_count", "char_count", "extraction_method"])
    if write_header:
        writer.writeheader()

    success = 0
    failed = 0

    for pdf_path in to_parse:
        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()

            # Try PyMuPDF text extraction first
            text = extract_text_pymupdf(pdf_path)
            method = "text"

            # If too little text, fall back to OCR
            if len(text.strip()) < ocr_threshold:
                print(f"  Low text yield for {pdf_path.name} ({len(text.strip())} chars), trying OCR...")
                ocr_text = extract_text_ocr(pdf_path)
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    method = "ocr"

            # Write extracted text
            out_path = TEXT_DIR / f"{pdf_path.stem}.txt"
            out_path.write_text(text, encoding="utf-8")

            writer.writerow({
                "filename": pdf_path.stem,
                "page_count": page_count,
                "char_count": len(text),
                "extraction_method": method,
            })
            meta_file.flush()
            success += 1

            if success % 100 == 0:
                print(f"  Parsed {success} PDFs...")

        except Exception as e:
            print(f"  Error parsing {pdf_path.name}: {e}")
            failed += 1

    meta_file.close()
    print(f"\nDone! Parsed: {success}, Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(description="Extract text from AAO decision PDFs")
    parser.add_argument("--ocr-threshold", type=int, default=100,
                        help="Min chars from PyMuPDF before falling back to OCR (default: 100)")
    args = parser.parse_args()
    parse_all(ocr_threshold=args.ocr_threshold)


if __name__ == "__main__":
    main()
