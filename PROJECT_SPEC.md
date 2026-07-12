# BookScribe - Project Specification

## Goal

Build a small Python command-line application that converts scanned book PDFs from open archives into text-based PDFs. The first version should read image-only PDF pages, run OCR on each page, and write the recognized text into a new PDF that is easier to scale and read on different devices.

The program must run as:

```bash
python main.py /path/to/file.pdf
```

## Scope

The first version focuses on a simple, working OCR pipeline:

1. Open a scanned PDF.
2. Render each page as an image.
3. Run OCR on each page image.
4. Write recognized text into a new text PDF.

The code should stay direct and practical: no unnecessary classes, no framework structure, and no abstractions until they are useful.

## OCR Choice

Use Unlimited-OCR as the default backend, with EasyOCR kept as a fallback backend.

Constraints:

- Keep the OCR reader isolated so backend changes do not rewrite the whole pipeline.
- Use the Unlimited-OCR multi-page path for page batches when possible.
- Use multiple Unlimited-OCR worker processes when requested or by default on the RTX 5090 machine; each worker loads one model copy.
- Keep EasyOCR English recognition available through `--ocr-backend easyocr`.

Note: Unlimited-OCR downloads `baidu/Unlimited-OCR` from Hugging Face on first use if it is not cached locally. EasyOCR may download its language model on first use if it is not already cached locally.

## Proposed File Structure

```text
BookScribe/
  main.py
  pdf_opener.py
  ocr.py
  font_classification.py  Classifies source text crops as regular/bold/italic/bold-italic
  pdf_writer.py
  PROJECT_SPEC.md
  requirements.txt
```

## Module Responsibilities

### `main.py`

Command-line entry point.

Responsibilities:

- Parse the input PDF path from `sys.argv`.
- Validate that the input file exists.
- Choose a default output path, for example `book_text.pdf` next to the input file.
- Call the PDF opener, OCR reader, and PDF writer in order.
- Print simple progress messages page by page.

### `pdf_opener.py`

PDF loading and page rendering.

Responsibilities:

- Open the source PDF.
- Render each page as an image suitable for OCR.
- Yield pages one at a time to avoid loading the whole book into memory.

Library: PyMuPDF (`fitz`), because it can open PDFs and render pages without requiring a separate Poppler installation.

### `ocr.py`

OCR backend module.

Responsibilities:

- Default to Unlimited-OCR with CUDA.
- Use EasyOCR with English only when `--ocr-backend easyocr` is selected.
- Default to `gpu=True`.
- Keep the EasyOCR reader cached inside a small reader object, not a module-level global.
- Provide one simple method, for example:

```python
def read_text_from_page(self, image) -> str:
    ...
```

- Return plain recognized text for a page.
- Hide OCR-backend-specific details from the rest of the program.

Future OCR backends can be added by extending this module while keeping the same method shape.

### `pdf_writer.py`

Text PDF creation.

Responsibilities:

- Create a new PDF containing selectable text.
- For Unlimited-OCR, use detected text and image coordinates to place text and cropped image regions.
- Segment source lines into words and use `font_classification.py` to choose regular, bold, italic, or bold-italic style for each word.
- Preserve source line positions and render adjacent words as selectable styled text.
- Preserve source line widths through source-derived interword spacing instead of reflowing paragraphs.
- Retain image-heavy or structurally complex source pages as raster pages with an invisible OCR layer when OCR tokens cannot reproduce their visible layout.
- Keep EasyOCR as a plain text fallback.

Library: PyMuPDF (`fitz`). Use it to create a new PDF, add pages, and insert OCR text directly.

### `font_classification.py`

Font-style classification.

Responsibilities:

- Train a small local CNN from installed serif book-font words and cache the weights after the first run.
- Predict bold and italic as independent word attributes, then calibrate them from page-relative slant and stroke evidence.
- Classify source word crops as regular, bold, italic, or bold-italic.
- Avoid regex-based attribution or author styling.

## Output Behavior

Default output path:

```text
<input_file_stem>_text.pdf
```

Example:

```bash
python main.py ~/Books/archive_scan.pdf
```

Creates:

```text
~/Books/archive_scan_text.pdf
```

## First-Version Limitations

- OCR quality depends on scan quality.
- Layout, columns, footnotes, illustrations, and page geometry will not be preserved exactly.
- The output PDF will prioritize readable text, not faithful visual reproduction.
- Only English OCR is enabled by default.
- No GUI in the first version.
- No batch folder processing in the first version.

## Dependencies

Initial dependencies:

```text
easyocr
PyMuPDF
numpy
```

`numpy` is used to convert rendered PyMuPDF pixmaps into image arrays that EasyOCR can read.

## First Implementation Milestone

The first milestone is complete when:

- `python main.py /path/to/file.pdf` works.
- The program creates `<input_file_stem>_text.pdf`.
- Each input page is OCR-processed in order.
- The output PDF contains selectable text.
- The code remains split into the four simple modules described above.
