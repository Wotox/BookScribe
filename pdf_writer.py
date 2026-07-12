import io
from pathlib import Path
import textwrap

import fitz
import numpy as np
from PIL import Image, ImageOps

from font_classification import FontClassifier


PAGE_WIDTH = 595
PAGE_HEIGHT = 842
MARGIN = 50
FONT_SIZE = 11
LINE_HEIGHT = 15
MAX_CHARS_PER_LINE = 86
COORD_SIZE = 1000


def write_text_pdf(page_texts, output_path):
    output_path = Path(output_path)
    document = fitz.open()

    try:
        for source_page_number, text in enumerate(page_texts, start=1):
            _add_text_pages(document, text, source_page_number)

        document.save(str(output_path))
    finally:
        document.close()


def write_unlimited_ocr_pdf(source_pdf_path, page_results, output_path):
    output_path = Path(output_path)
    font_classifier = FontClassifier()
    source = fitz.open(str(source_pdf_path))
    target = fitz.open()

    try:
        for page_result in page_results:
            source_page = source[page_result["page_number"] - 1]
            output_page = target.new_page(
                width=source_page.rect.width,
                height=source_page.rect.height,
            )
            blocks = parse_unlimited_ocr_blocks(page_result["text"])
            _add_unlimited_ocr_blocks(output_page, page_result, blocks, font_classifier)

        target.save(str(output_path), garbage=4, deflate=True)
    finally:
        target.close()
        source.close()


def parse_unlimited_ocr_blocks(text):
    blocks = []
    position = 0
    start_marker = "<|det|>"
    end_marker = "<|/det|>"

    while True:
        start = text.find(start_marker, position)
        if start == -1:
            break

        metadata_start = start + len(start_marker)
        metadata_end = text.find(end_marker, metadata_start)
        if metadata_end == -1:
            break

        metadata = text[metadata_start:metadata_end].strip()
        content_start = metadata_end + len(end_marker)
        next_start = text.find(start_marker, content_start)
        content_end = len(text) if next_start == -1 else next_start
        kind, bbox = _parse_detection_metadata(metadata)
        blocks.append(
            {
                "kind": kind,
                "bbox": bbox,
                "text": _normalize_pdf_text(text[content_start:content_end].strip()),
            }
        )
        position = content_end

    if blocks:
        return blocks

    stripped = _normalize_pdf_text(text.strip())
    if not stripped:
        return []

    return [{"kind": "text", "bbox": [0, 0, COORD_SIZE, COORD_SIZE], "text": stripped}]


def _parse_detection_metadata(metadata):
    left = metadata.find("[")
    right = metadata.find("]", left)
    if left == -1 or right == -1:
        raise ValueError(f"Invalid UnlimitedOCR detection metadata: {metadata}")

    kind = metadata[:left].strip().split()[0]
    bbox = [float(part.strip()) for part in metadata[left + 1 : right].split(",")]
    if len(bbox) != 4:
        raise ValueError(f"Expected 4 bbox values in: {metadata}")

    return kind, bbox


def _add_unlimited_ocr_blocks(page, page_result, blocks, font_classifier):
    with Image.open(page_result["image_path"]) as image:
        source_image = image.convert("RGB")

        for block in blocks:
            rect = _scale_bbox(block["bbox"], page.rect)
            if rect.is_empty or rect.width <= 1 or rect.height <= 1:
                continue

            if block["kind"] == "image":
                crop = _crop_source_region(source_image, block["bbox"])
                page.insert_image(rect, stream=_image_bytes(crop))
                continue

            text = block["text"].strip()
            if not text:
                continue

            crop = _crop_source_region(source_image, block["bbox"])
            _insert_classified_text(page, rect, text, crop, block["kind"], font_classifier)


def _insert_classified_text(page, rect, text, crop, kind, font_classifier):
    if kind == "title":
        base_choice = font_classifier.choices["bold"]
    else:
        base_choice = font_classifier.classify(crop)
    font_size, lines = _fit_lines(text, rect, base_choice, kind, font_classifier)
    line_crops = _line_crops_from_image(crop, len(lines))
    line_choices = _line_font_choices(font_classifier, line_crops, base_choice, len(lines), kind)

    y = rect.y0 + font_size

    for line, choice in zip(lines, line_choices):
        x = rect.x0
        page.insert_text(
            (x, y),
            line,
            fontsize=font_size,
            **font_classifier.insert_kwargs(choice),
            color=(0, 0, 0),
        )
        y += _line_height(font_size)


def _line_font_choices(font_classifier, line_crops, base_choice, line_count, kind):
    if line_crops is None:
        return [base_choice] * line_count

    choices = [font_classifier.classify(line_crop) for line_crop in line_crops]
    if kind == "text" and line_count > 1:
        return _smooth_multiline_text_choices(font_classifier, choices)

    italic_count = sum(choice.style in ("italic", "bold_italic") for choice in choices)
    if (
        line_count > 2
        and base_choice.style in ("regular", "bold")
        and italic_count / line_count > 0.4
    ):
        return [base_choice] * line_count

    return choices


def _smooth_multiline_text_choices(font_classifier, choices):
    counts = {}
    for choice in choices:
        counts[choice.style] = counts.get(choice.style, 0) + 1

    style, count = max(counts.items(), key=lambda item: item[1])
    share = count / len(choices)
    if style == "regular" or share >= 0.75:
        return [font_classifier.choices[style]] * len(choices)

    return [font_classifier.choices["regular"]] * len(choices)


def _fit_lines(text, rect, base_choice, kind, font_classifier):
    max_size = 24 if kind == "title" else 12
    min_size = 4.0
    size = min(max_size, max(min_size, rect.height * 0.72))

    while size >= min_size:
        lines = _wrap_text_for_width(text, rect.width, base_choice, size, font_classifier)
        if len(lines) * _line_height(size) <= rect.height + 0.01:
            return size, lines
        size -= 0.25

    return min_size, _wrap_text_for_width(text, rect.width, base_choice, min_size, font_classifier)


def _wrap_text_for_width(text, max_width, font_choice, font_size, font_classifier):
    words = text.split()
    if not words:
        return [""]

    lines = []
    line = words[0]

    for word in words[1:]:
        candidate = f"{line} {word}"
        if font_classifier.measure_width(candidate, font_choice, font_size) <= max_width:
            line = candidate
        else:
            lines.append(line)
            line = word

    lines.append(line)
    return lines


def _line_crops_from_image(crop, expected_count):
    if expected_count <= 1:
        return [crop]

    gray = ImageOps.grayscale(crop)
    gray = ImageOps.autocontrast(gray)
    array = np.asarray(gray, dtype=np.uint8)
    threshold = min(235, max(35, int(array.mean() - array.std() * 0.15)))
    ink = array < threshold
    row_counts = ink.sum(axis=1)
    if int(row_counts.max()) <= 0:
        return None

    active = row_counts > max(2, row_counts.max() * 0.08)
    active = _dilate_rows(active, 2)
    bands = _row_bands(active)
    bands = [(top, bottom) for top, bottom in bands if bottom - top >= 2]
    bands = _merge_bands_to_count(bands, expected_count)

    if len(bands) != expected_count:
        return None

    return [crop.crop((0, top, crop.width, bottom)) for top, bottom in bands]


def _dilate_rows(active, radius):
    result = active.copy()
    for offset in range(1, radius + 1):
        result[offset:] |= active[:-offset]
        result[:-offset] |= active[offset:]
    return result


def _row_bands(active):
    bands = []
    start = None

    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            bands.append((start, index))
            start = None

    if start is not None:
        bands.append((start, len(active)))

    return bands


def _merge_bands_to_count(bands, expected_count):
    bands = list(bands)
    while len(bands) > expected_count:
        gaps = [
            (bands[index + 1][0] - bands[index][1], index)
            for index in range(len(bands) - 1)
        ]
        _gap, index = min(gaps)
        merged = (bands[index][0], bands[index + 1][1])
        bands = bands[:index] + [merged] + bands[index + 2 :]
    return bands


def _crop_source_region(source_image, bbox):
    x0, y0, x1, y1 = bbox
    crop_box = (
        _clamp(round((x0 / COORD_SIZE) * source_image.width), 0, source_image.width),
        _clamp(round((y0 / COORD_SIZE) * source_image.height), 0, source_image.height),
        _clamp(round((x1 / COORD_SIZE) * source_image.width), 0, source_image.width),
        _clamp(round((y1 / COORD_SIZE) * source_image.height), 0, source_image.height),
    )
    return source_image.crop(crop_box)


def _image_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _scale_bbox(bbox, page_rect):
    x0, y0, x1, y1 = bbox
    return fitz.Rect(
        page_rect.x0 + (x0 / COORD_SIZE) * page_rect.width,
        page_rect.y0 + (y0 / COORD_SIZE) * page_rect.height,
        page_rect.x0 + (x1 / COORD_SIZE) * page_rect.width,
        page_rect.y0 + (y1 / COORD_SIZE) * page_rect.height,
    )


def _line_height(font_size):
    return font_size * 1.14


def _normalize_pdf_text(text):
    try:
        text = text.encode("latin1").decode("utf-8")
    except UnicodeError:
        text = (
            text.replace("\u00e2\u20ac\u201d", "\u2014")
            .replace("\u00e2\u20ac\u201c", "\u2013")
            .replace("\u00e2\u20ac\u02dc", "\u2018")
            .replace("\u00e2\u20ac\u2122", "\u2019")
            .replace("\u00e2\u20ac\u0153", "\u201c")
            .replace("\u00e2\u20ac\u009d", "\u201d")
            .replace("\u00c2", "")
        )

    return (
        text.replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def _clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


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
