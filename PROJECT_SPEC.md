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

## Initial OCR Choice

Use EasyOCR with English recognition for the first implementation.

Constraints:

- Do not download large OCR models.
- Use only the EasyOCR English model for the initial version.
- Keep the OCR reader isolated so another OCR backend can be added later without rewriting the whole pipeline.

Note: EasyOCR may download its language model on first use if it is not already cached locally. The program should document this clearly and avoid requesting extra languages by default.

## Proposed File Structure

```text
BookScribe/
  main.py
  pdf_opener.py
  ocr.py
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

- Create the EasyOCR reader with English only.
- Default to EasyOCR `gpu=True`; EasyOCR uses CUDA or Apple MPS when available and falls back to CPU otherwise.
- Keep the EasyOCR reader cached inside a small reader object, not a module-level global.
- Provide one simple method, for example:

```python
def read_text_from_page(self, image) -> str:
    ...
```

- Return plain recognized text for a page.
- Hide EasyOCR-specific details from the rest of the program.

Future OCR backends can be added by replacing or extending this module while keeping the same method shape.

### `pdf_writer.py`

Text PDF creation.

Responsibilities:

- Create a new PDF containing selectable text.
- Add one output page per source page.
- Write OCR text with readable margins and font size.
- Handle long text by wrapping lines and continuing onto additional pages if needed.

Library: PyMuPDF (`fitz`). Use it to create a new PDF, add pages, and insert OCR text directly.

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
