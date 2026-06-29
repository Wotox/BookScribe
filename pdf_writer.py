from pathlib import Path
import textwrap

import fitz


PAGE_WIDTH = 595
PAGE_HEIGHT = 842
MARGIN = 50
FONT_SIZE = 11
LINE_HEIGHT = 15
MAX_CHARS_PER_LINE = 86


def write_text_pdf(page_texts, output_path):
    output_path = Path(output_path)
    document = fitz.open()

    try:
        for source_page_number, text in enumerate(page_texts, start=1):
            _add_text_pages(document, text, source_page_number)

        document.save(str(output_path))
    finally:
        document.close()


def _add_text_pages(document, text, source_page_number):
    lines = _wrap_page_text(text)
    if not lines:
        lines = ["[No text recognized]"]

    page = None
    y = 0

    for line in lines:
        if page is None or y > PAGE_HEIGHT - MARGIN:
            page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
            y = MARGIN
            heading = f"Source page {source_page_number}"
            page.insert_text((MARGIN, y), heading, fontsize=9, fontname="helv")
            y += LINE_HEIGHT * 2

        page.insert_text((MARGIN, y), line, fontsize=FONT_SIZE, fontname="helv")
        y += LINE_HEIGHT


def _wrap_page_text(text):
    lines = []

    for paragraph in text.splitlines():
        paragraph = paragraph.strip()

        if not paragraph:
            lines.append("")
            continue

        lines.extend(textwrap.wrap(paragraph, width=MAX_CHARS_PER_LINE))

    return lines
