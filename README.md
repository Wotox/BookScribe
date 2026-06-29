# BookScribe

BookScribe converts scanned book PDFs into text-based PDFs using OCR.

It is built for plain archive scans: image-only PDFs where each page is a scan of a book page. The output PDF prioritizes selectable, scalable text over preserving the original page layout.

## Current Status

This is an early working version. OCR can be messy, and the generated PDF does not yet recreate the original styling, columns, fonts, or exact page structure.

The first goal is simple:

```text
scanned PDF -> page images -> OCR text -> text-only PDF
```

## Usage

From the project folder:

```bash
source .venv/bin/activate
python main.py /path/to/file.pdf
```

The output is created next to the input file:

```text
<input_file_stem>_text.pdf
```

## Setup

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The current dependencies are:

```text
easyocr
PyMuPDF
numpy
```

Note: EasyOCR may download its English recognition model the first time it runs if the model is not already cached locally.

## Project Structure

```text
main.py        CLI entry point
pdf_opener.py  Opens PDFs and renders pages as images
ocr.py         Runs EasyOCR on page images
pdf_writer.py  Writes recognized text into a new PDF
```

## Limitations

- English OCR only by default.
- OCR quality depends heavily on scan quality.
- Original book layout is not preserved.
- Text styling is currently basic.
- No GUI.
- No batch folder processing yet.
