# BookScribe

BookScribe converts scanned book PDFs into text-based PDFs using OCR.

It is built for plain archive scans: image-only PDFs where each page is a scan of a book page. The default Unlimited-OCR path reconstructs a text PDF using OCR coordinates, cropped image blocks, and visual font-style classification.

## Current Status

This is an early working version. OCR can be messy, and font recovery is approximate because Unlimited-OCR does not report the original font metadata.

The first goal is simple:

```text
scanned PDF -> page images -> Unlimited-OCR blocks -> font classification -> reconstructed text PDF
```

## Usage

From the project folder:

```powershell
.\.venv\Scripts\Activate.ps1
python .\main.py C:\path\to\file.pdf
```

Unlimited-OCR is the default OCR backend. It uses text and image coordinates from the model, classifies visible source text crops with the small CNN in `font_classification.py`, and writes a reconstructed text PDF.

The font-style classifier is trained locally from installed serif book fonts and cached under `__pycache__` after the first run. It does not use author-name or attribution regexes.

To use the older EasyOCR backend:

```powershell
python .\main.py --ocr-backend easyocr C:\path\to\file.pdf
```

Unlimited-OCR processes 4 pages per model call and uses 2 worker processes by default. Each worker loads its own model copy, which improves GPU utilization on large GPUs but uses more VRAM.

Tune the worker count first:

```powershell
python .\main.py --ocr-workers 1 C:\path\to\file.pdf
python .\main.py --ocr-workers 2 C:\path\to\file.pdf
```

On a large GPU, you can also try a larger page chunk:

```powershell
python .\main.py --ocr-batch-size 8 C:\path\to\file.pdf
```

To test a subset of pages:

```powershell
python .\main.py C:\path\to\file.pdf --pages 1-20
```

The output is created next to the input file:

```text
<input_file_stem>_text.pdf
<input_file_stem>_pages_001_020_text.pdf  when --pages 1-20 is used
```

## OCR Comparison

To compare OCR backends on pages 1, 3, and 5 of the Lee Kuan Yew test book:

```powershell
python .\compare_ocr.py --pages 1,3,5
```

The script writes rendered page images, per-backend Markdown/text outputs, `result.json` files, `summary.json`, and a readable `results_by_page.md` under:

```text
ocr_comparison\<timestamp>\
```

For normal inspection, open the stable latest report instead of browsing date folders:

```text
ocr_comparison\latest_results.md
```

That report is grouped by page, then by OCR backend, and includes the exact text each model extracted.

Backends currently included:

```text
easyocr
unlimited-ocr
surya-ocr
paddleocr-vl
nuextract3
```

PaddleOCR-VL and NuExtract3 need Transformers v5, while Unlimited-OCR currently needs Transformers 4.57.1. Keep the main `.venv` on `requirements.txt`, then install the v5 runtime into the ignored vendor folder:

```powershell
python -m pip install --target .\.ocr_vendor\transformers5 -r .\requirements-ocr-transformers5.txt
```

`compare_ocr.py` automatically uses that vendored Transformers v5 runtime only for PaddleOCR-VL and NuExtract3 workers.

Surya OCR 2 needs an inference backend. On this Windows machine, the comparison script uses the ignored llama.cpp binary at:

```text
.ocr_vendor\llama.cpp\b9843\llama-server.exe
```

If that file is missing, Surya falls back to its default backend selection, which may require Docker Desktop or a manually configured `LLAMA_CPP_BINARY`.

## Setup

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The current dependencies are:

```text
torch
torchvision
transformers
accelerate
addict
easydict
einops
matplotlib
requests
tqdm
surya-ocr
easyocr
PyMuPDF
Pillow
numpy
```

Note: Unlimited-OCR requires CUDA-enabled PyTorch and downloads `baidu/Unlimited-OCR` from Hugging Face on first use if it is not already cached locally. The official model loader uses Hugging Face `trust_remote_code=True`, so the first real run executes custom model code from that repository. EasyOCR may also download its English recognition model the first time it runs.

## Project Structure

```text
main.py        CLI entry point
pdf_opener.py  Opens PDFs and renders pages as images
ocr.py         Runs the selected OCR backend on page images
font_classification.py  Classifies source text crops as regular/bold/italic/bold-italic
pdf_writer.py  Writes recognized text and cropped images into a new PDF
```

## Limitations

- English OCR only by default.
- OCR quality depends heavily on scan quality.
- Original book layout is not preserved.
- Text styling is currently basic.
- No GUI.
- No batch folder processing yet.
