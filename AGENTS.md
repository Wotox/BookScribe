# Agent Notes

BookScribe is intentionally small and direct. Do not add unnecessary classes, frameworks, or abstractions.

## Current Architecture

The project is split into simple modules:

```text
main.py        CLI entry point
pdf_opener.py  Opens PDFs and renders pages as images
ocr.py         Runs the selected OCR backend on page images
font_classification.py  Classifies source word crops as regular/bold/italic/bold-italic
pdf_writer.py  Writes recognized text and cropped images into a new PDF
```

PyMuPDF is used for both sides of PDF work:

- rendering scanned source pages for OCR
- creating the final text PDF

OCR backend details are isolated in `ocr.py` so the rest of the pipeline can stay unchanged.

Font style recovery is isolated in `font_classification.py`. It uses a small two-attribute CNN trained from local serif book-font words, plus page-relative slant and stroke calibration, and caches the trained weights under `__pycache__`; do not reintroduce regex-based author or attribution styling.

`pdf_writer.py` segments each Unlimited-OCR text block into source lines and word boxes, aligns OCR tokens in reading order, and renders each word with its visually classified style. Keep mixed styles within a line and preserve the detected source-line position.

Body lines use source-derived font size and justified spacing so their starts and ends match the scan. Reconstruct detected image blocks normally. Preserve source ink groups that cannot be aligned to OCR tokens as transparent residual graphics; reserve full-page raster preservation for low-text color pages.

`OCRReader` should default to `unlimited-ocr` using `baidu/Unlimited-OCR` from Hugging Face. Keep EasyOCR available as the fallback backend via `--ocr-backend easyocr`.

`ocr.py` uses a small `OCRReader` class to cache the expensive OCR model/reader instance without module-level mutable globals. Keep that class narrow.

Unlimited-OCR should use its multi-page `infer_multi()` path for batches larger than one page. The CLI default batch size is 4 pages; keep that conservative unless the user asks to tune for a specific GPU.

Unlimited-OCR uses 2 worker processes by default on this machine so the RTX 5090 is fed by two model instances. Each worker loads its own model copy, so reduce to `--ocr-workers 1` if VRAM gets tight.

`OCRReader` should default to `gpu=True`. Unlimited-OCR currently requires CUDA in its Hugging Face implementation. In EasyOCR 1.7.2, `gpu=True` tries CUDA first, then Apple MPS, then falls back to CPU with a warning if neither backend is available.

## Local Setup Notes

Create the local venv normally:

```bash
python -m venv .venv
```

Keep dependencies installed inside `.venv`. Avoid relying on system or user site-packages for OCR/PyTorch because mixed PyTorch environments are fragile.

## OCR Comparison Notes

`compare_ocr.py` tests OCR backends on selected pages and writes ignored outputs under `ocr_comparison/`.

Keep the main `.venv` on Transformers 4.57.1 for Unlimited-OCR compatibility. PaddleOCR-VL and NuExtract3 need Transformers v5, so the comparison script uses the ignored `.ocr_vendor/transformers5` target directory for only those worker subprocesses. Recreate it with:

```bash
python -m pip install --target ./.ocr_vendor/transformers5 -r requirements-ocr-transformers5.txt
```

Surya OCR 2 uses llama.cpp on Windows when `.ocr_vendor/llama.cpp/b9843/llama-server.exe` exists. Without that binary it may try Docker/vLLM and fail if Docker Desktop's Linux engine is not running.
