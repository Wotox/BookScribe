# Agent Notes

BookScribe is intentionally small and direct. Do not add unnecessary classes, frameworks, or abstractions.

## Current Architecture

The project is split into simple modules:

```text
main.py        CLI entry point
pdf_opener.py  Opens PDFs and renders pages as images
ocr.py         Runs EasyOCR on page images
pdf_writer.py  Writes recognized text into a new PDF
```

PyMuPDF is used for both sides of PDF work:

- rendering scanned source pages for OCR
- creating the final text PDF

EasyOCR is isolated in `ocr.py` so another OCR backend can be added later without rewriting the whole pipeline.

`ocr.py` uses a small `OCRReader` class to cache the expensive EasyOCR reader instance without module-level mutable globals. Keep that class narrow.

`OCRReader` should default to `gpu=True`. In EasyOCR 1.7.2 this tries CUDA first, then Apple MPS, then falls back to CPU with a warning if neither backend is available.

## Local Setup Notes

The user's current venv was created with:

```bash
python3 -m venv --system-site-packages .venv
```

That was intentional because EasyOCR and related ML dependencies were already available in the system Python environment, and the user wanted to avoid unnecessary traffic and large downloads.

PyMuPDF was the only missing dependency during initial setup and was installed into `.venv`.

Avoid reinstalling EasyOCR, PyTorch, or OCR models unless the user explicitly asks. EasyOCR may download the English recognition model on first real OCR use if it is not cached locally.
