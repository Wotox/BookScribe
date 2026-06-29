from pathlib import Path
import sys

from ocr import OCRReader
from pdf_opener import render_pages
from pdf_writer import write_text_pdf


def output_path_for(input_path):
    return input_path.with_name(f"{input_path.stem}_text.pdf")


def main():
    if len(sys.argv) != 2:
        print("BookScribe")
        print("Usage: python main.py /path/to/file.pdf")
        return 2

    input_path = Path(sys.argv[1]).expanduser()
    if not input_path.is_file():
        print("BookScribe")
        print(f"Input file not found: {input_path}")
        return 1

    print("BookScribe")
    output_path = output_path_for(input_path)
    page_texts = []
    ocr_reader = OCRReader()

    for page_number, image in render_pages(input_path):
        print(f"OCR page {page_number}...")
        page_texts.append(ocr_reader.read_text_from_page(image))

    write_text_pdf(page_texts, output_path)
    print(f"Created: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
