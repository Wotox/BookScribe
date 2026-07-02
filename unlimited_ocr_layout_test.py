import argparse
import json
import re
from pathlib import Path

import fitz
from PIL import Image

from ocr import DEFAULT_BATCH_SIZE, OCRReader


DEFAULT_PDF = Path(
    r"C:\Users\hplus\AI_Dev\bookscribe\Scanned"
    r"\The_Singapore_story_memoirs_of_Lee_Kuan_Yew_Lee,_Kuan_Yew,_Kuan.pdf"
)
DEFAULT_PAGES = "1,3,5"
DEFAULT_OUTPUT_DIR = Path("ocr_comparison") / "unlimited_ocr_layout"
COORD_SIZE = 1000
FONT_DIR = Path(r"C:\Windows\Fonts")
DEFAULT_ZOOM = 2.0

DETECTION_PATTERN = re.compile(
    r"<\|det\|>(?P<kind>[A-Za-z0-9_-]+)\s*"
    r"\[(?P<bbox>[^\]]+)\]<\|/det\|>"
    r"(?P<text>.*?)(?=\n?<\|det\|>|\Z)",
    re.DOTALL,
)


def main():
    args = parse_args()
    page_numbers = parse_pages(args.pages)
    output_dir = args.output_dir
    names = output_names(page_numbers)
    paths = output_paths(output_dir, names)
    fonts = load_font_policy(args.font_family)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rendering source pages {format_page_list(page_numbers)}", flush=True)
    rendered_pages = render_pages(args.pdf, page_numbers, paths["source_pages"], args.zoom)

    page_texts = read_or_run_ocr(
        rendered_pages,
        paths["ocr_text"],
        args.ocr_text_dir,
        args.ocr_batch_size,
        args.force_ocr,
    )
    blocks_by_page = parse_and_write_blocks(page_numbers, page_texts, paths["blocks"])

    write_reconstructed_pdf(
        args.pdf,
        page_numbers,
        blocks_by_page,
        rendered_pages,
        paths["images"],
        paths["text_pdf"],
        fonts,
        debug=False,
    )
    write_reconstructed_pdf(
        args.pdf,
        page_numbers,
        blocks_by_page,
        rendered_pages,
        paths["images"],
        paths["text_debug_pdf"],
        fonts,
        debug=True,
    )

    render_pdf_pages(paths["text_pdf"], paths["reconstructed_pages"], "page", page_numbers, args.compare_zoom)
    render_comparisons(paths["source_pages"], paths["reconstructed_pages"], paths["comparison"])

    summary = {
        "pdf": str(args.pdf),
        "pages": page_numbers,
        "coord_size": COORD_SIZE,
        "font_family": fonts["family"],
        "font_files": {style: str(font["fontfile"]) for style, font in fonts["styles"].items()},
        "ocr_batch_size": args.ocr_batch_size,
        "rendered_pages": rendered_pages,
        "text_pdf": str(paths["text_pdf"]),
        "text_debug_pdf": str(paths["text_debug_pdf"]),
        "ocr_text_dir": str(paths["ocr_text"]),
        "block_dir": str(paths["blocks"]),
        "image_dir": str(paths["images"]),
        "original_pages_dir": str(paths["source_pages"]),
        "reconstructed_pages_dir": str(paths["reconstructed_pages"]),
        "comparison_dir": str(paths["comparison"]),
    }
    write_json(output_dir / "summary.json", summary)

    print(f"Text PDF: {paths['text_pdf']}")
    print(f"Text debug PDF: {paths['text_debug_pdf']}")
    print(f"Raw OCR: {paths['ocr_text']}")
    print(f"Blocks: {paths['blocks']}")
    print(f"Images: {paths['images']}")
    print(f"Original pages: {paths['source_pages']}")
    print(f"Reconstructed pages: {paths['reconstructed_pages']}")
    print(f"Comparison pages: {paths['comparison']}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a reconstructed text PDF from Unlimited-OCR coordinates."
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--pages", default=DEFAULT_PAGES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zoom", type=float, default=DEFAULT_ZOOM)
    parser.add_argument("--compare-zoom", type=float, default=1.5)
    parser.add_argument("--ocr-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--force-ocr", action="store_true")
    parser.add_argument(
        "--font-family",
        choices=("times",),
        default="times",
        help="Generic book font policy. Default: times.",
    )
    parser.add_argument(
        "--ocr-text-dir",
        type=Path,
        default=None,
        help="Optional folder containing existing Unlimited-OCR page_###.md outputs.",
    )
    return parser.parse_args()


def output_names(page_numbers):
    return f"pages_{page_numbers[0]:03}_{page_numbers[-1]:03}"


def output_paths(output_dir, names):
    return {
        "text_pdf": output_dir / f"unlimited_ocr_{names}_text.pdf",
        "text_debug_pdf": output_dir / f"unlimited_ocr_{names}_text_debug.pdf",
        "ocr_text": output_dir / "unlimited-ocr",
        "blocks": output_dir / "blocks",
        "images": output_dir / "images",
        "source_pages": output_dir / "original_pages",
        "reconstructed_pages": output_dir / "reconstructed_pages",
        "comparison": output_dir / "comparison",
    }


def render_pages(pdf_path, page_numbers, output_dir, zoom):
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_pages = []
    document = fitz.open(str(pdf_path))

    try:
        matrix = fitz.Matrix(zoom, zoom)
        for page_number in page_numbers:
            source_page = document[page_number - 1]
            pixmap = source_page.get_pixmap(matrix=matrix, alpha=False)
            image_path = output_dir / f"page_{page_number:03}.png"
            pixmap.save(str(image_path))
            rendered_pages.append(
                {
                    "page": page_number,
                    "image_file": str(image_path),
                    "width": pixmap.width,
                    "height": pixmap.height,
                }
            )
            print(f"Rendered page {page_number}", flush=True)
    finally:
        document.close()

    return rendered_pages


def read_or_run_ocr(rendered_pages, output_dir, source_text_dir, batch_size, force_ocr):
    output_dir.mkdir(parents=True, exist_ok=True)
    page_texts = {}
    missing_pages = []

    for page in rendered_pages:
        page_number = page["page"]
        output_path = output_dir / f"page_{page_number:03}.md"
        source_path = (
            source_text_dir / f"page_{page_number:03}.md" if source_text_dir is not None else None
        )

        if not force_ocr and output_path.is_file():
            page_texts[page_number] = output_path.read_text(encoding="utf-8").strip()
            print(f"OCR page {page_number}: cached", flush=True)
        elif not force_ocr and source_path is not None and source_path.is_file():
            text = source_path.read_text(encoding="utf-8").strip()
            output_path.write_text(text, encoding="utf-8")
            page_texts[page_number] = text
            print(f"OCR page {page_number}: copied from {source_text_dir}", flush=True)
        else:
            missing_pages.append(page)

    if missing_pages:
        reader = OCRReader(backend="unlimited-ocr")
        for batch in batches(missing_pages, batch_size):
            page_numbers = [page["page"] for page in batch]
            image_paths = [page["image_file"] for page in batch]
            print(f"OCR pages {format_page_list(page_numbers)}", flush=True)
            texts = reader.read_text_from_image_paths(image_paths)

            if len(texts) != len(batch):
                raise RuntimeError(
                    f"Unlimited-OCR returned {len(texts)} pages for {len(batch)} inputs."
                )

            for page, text in zip(batch, texts):
                page_number = page["page"]
                output_path = output_dir / f"page_{page_number:03}.md"
                text = text.strip()
                output_path.write_text(text, encoding="utf-8")
                page_texts[page_number] = text
                print(f"OCR page {page_number}: processed", flush=True)

    return page_texts


def parse_and_write_blocks(page_numbers, page_texts, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks_by_page = {}

    for page_number in page_numbers:
        blocks = parse_unlimited_ocr_blocks(page_texts[page_number])
        blocks_by_page[page_number] = blocks
        write_json(output_dir / f"page_{page_number:03}.json", blocks)
        print(f"Blocks page {page_number}: {len(blocks)}", flush=True)

    return blocks_by_page


def parse_unlimited_ocr_blocks(text):
    blocks = []

    for match in DETECTION_PATTERN.finditer(text):
        bbox = parse_bbox(match.group("bbox"))
        blocks.append(
            {
                "kind": match.group("kind"),
                "bbox": bbox,
                "text": repair_text(match.group("text").strip()),
            }
        )

    if blocks:
        return blocks

    return [{"kind": "text", "bbox": [0, 0, COORD_SIZE, COORD_SIZE], "text": repair_text(text.strip())}]


def parse_bbox(value):
    numbers = [float(part.strip()) for part in value.split(",")]
    if len(numbers) != 4:
        raise ValueError(f"Expected 4 bbox values, got {len(numbers)}: {value}")

    return numbers


def write_reconstructed_pdf(
    pdf_path,
    page_numbers,
    blocks_by_page,
    rendered_pages,
    image_dir,
    output_path,
    fonts,
    debug,
):
    source = fitz.open(str(pdf_path))
    target = fitz.open()
    rendered_by_page = {page["page"]: page for page in rendered_pages}
    image_dir.mkdir(parents=True, exist_ok=True)

    try:
        for page_number in page_numbers:
            source_page = source[page_number - 1]
            target_page = target.new_page(
                width=source_page.rect.width,
                height=source_page.rect.height,
            )
            add_reconstructed_blocks_to_page(
                target_page,
                blocks_by_page[page_number],
                rendered_by_page[page_number],
                image_dir,
                page_number,
                fonts,
                debug,
            )

        target.save(str(output_path), garbage=4, deflate=True)
    finally:
        target.close()
        source.close()


def add_reconstructed_blocks_to_page(
    page,
    blocks,
    rendered_page,
    image_dir,
    page_number,
    fonts,
    debug,
):
    image_index = 0

    for block in blocks:
        rect = scale_bbox(block["bbox"], page.rect)
        kind = block["kind"]

        if kind == "image":
            image_index += 1
            image_path = crop_image_block(
                rendered_page,
                block["bbox"],
                image_dir / f"page_{page_number:03}_image_{image_index:02}.png",
            )
            page.insert_image(rect, filename=str(image_path))
            if debug:
                draw_debug_rect(page, rect, kind)
            continue

        if debug:
            draw_debug_rect(page, rect, kind)

        text = block["text"].strip()
        if not text:
            continue

        segments = style_segments_for_block(text, kind, fonts)
        font_size, lines = fit_segments_to_rect(rect, segments)
        insert_visible_lines(page, rect, lines, font_size)


def crop_image_block(rendered_page, bbox, output_path):
    image = Image.open(rendered_page["image_file"])
    x0, y0, x1, y1 = bbox
    crop_box = (
        clamp(round((x0 / COORD_SIZE) * rendered_page["width"]), 0, rendered_page["width"]),
        clamp(round((y0 / COORD_SIZE) * rendered_page["height"]), 0, rendered_page["height"]),
        clamp(round((x1 / COORD_SIZE) * rendered_page["width"]), 0, rendered_page["width"]),
        clamp(round((y1 / COORD_SIZE) * rendered_page["height"]), 0, rendered_page["height"]),
    )
    cropped = image.crop(crop_box)
    cropped.save(output_path)
    return output_path


def scale_bbox(bbox, page_rect):
    x0, y0, x1, y1 = bbox
    return fitz.Rect(
        page_rect.x0 + (x0 / COORD_SIZE) * page_rect.width,
        page_rect.y0 + (y0 / COORD_SIZE) * page_rect.height,
        page_rect.x0 + (x1 / COORD_SIZE) * page_rect.width,
        page_rect.y0 + (y1 / COORD_SIZE) * page_rect.height,
    )


def style_segments_for_block(text, kind, fonts):
    base_style = "bold" if kind == "title" else "regular"

    if is_attribution(text):
        return [{"text": text, "font": fonts["styles"]["italic"], "align": "center"}]

    attribution_match = re.search(r"\s-\s(?=[A-Z][A-Za-z .'-]+,\s)", text)
    if attribution_match:
        body = text[: attribution_match.start()].strip()
        attribution = text[attribution_match.start() :].strip()
        segments = []
        if body:
            segments.append({"text": body, "font": fonts["styles"][base_style], "align": "left"})
        if attribution:
            segments.append({"text": attribution, "font": fonts["styles"]["italic"], "align": "right"})
        return segments

    return [{"text": text, "font": fonts["styles"][base_style], "align": "left"}]


def is_attribution(text):
    stripped = text.strip()
    return stripped.startswith("- ") and len(stripped.split()) >= 3


def fit_segments_to_rect(rect, segments):
    kind_is_title = any(segment["font"]["style"] == "bold" for segment in segments)
    max_size = 24 if kind_is_title else 12
    min_size = 4.0
    size = min(max_size, max(min_size, rect.height * 0.72))

    while size >= min_size:
        lines = wrap_segments_for_rect(segments, rect.width, size)
        if len(lines) * line_height(size) <= rect.height + 0.01:
            return size, lines
        size -= 0.25

    return min_size, wrap_segments_for_rect(segments, rect.width, min_size)


def wrap_segments_for_rect(segments, max_width, font_size):
    lines = []

    for segment in segments:
        segment_lines = wrap_text_for_width(segment["text"], segment["font"], max_width, font_size)
        lines.extend({"text": line, "font": segment["font"], "align": segment["align"]} for line in segment_lines)

    return lines or [{"text": "", "font": segments[0]["font"], "align": "left"}]


def wrap_text_for_width(text, font, max_width, font_size):
    words = text.split()
    if not words:
        return [""]

    lines = []
    line = words[0]

    for word in words[1:]:
        candidate = f"{line} {word}"
        if text_width(candidate, font, font_size) <= max_width:
            line = candidate
        else:
            lines.append(line)
            line = word

    lines.append(line)
    return lines


def insert_visible_lines(page, rect, lines, font_size):
    y = rect.y0 + font_size

    for line in lines:
        font = line["font"]
        x = aligned_x(rect, line["text"], font, font_size, line["align"])
        page.insert_text(
            (x, y),
            line["text"],
            fontsize=font_size,
            fontname=font["name"],
            fontfile=str(font["fontfile"]),
            set_simple=1,
            color=(0, 0, 0),
        )
        y += line_height(font_size)


def aligned_x(rect, text, font, font_size, align):
    if align == "center":
        return rect.x0 + max(0, (rect.width - text_width(text, font, font_size)) / 2)
    if align == "right":
        return rect.x1 - text_width(text, font, font_size)
    return rect.x0


def text_width(text, font, font_size):
    return font["measure"].text_length(text, fontsize=font_size)


def line_height(font_size):
    return font_size * 1.14


def draw_debug_rect(page, rect, kind):
    colors = {
        "title": (1, 0, 0),
        "text": (0, 0.2, 1),
        "image": (0, 0.6, 0),
    }
    page.draw_rect(rect, color=colors.get(kind, (0.8, 0, 0.8)), width=0.7)


def render_pdf_pages(pdf_path, output_dir, prefix, page_numbers, zoom):
    output_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(str(pdf_path))

    try:
        matrix = fitz.Matrix(zoom, zoom)
        for page_index, page in enumerate(document):
            source_page_number = page_numbers[page_index]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            pixmap.save(str(output_dir / f"{prefix}_{source_page_number:03}.png"))
            print(f"Rendered reconstructed page {source_page_number}", flush=True)
    finally:
        document.close()


def render_comparisons(source_dir, reconstructed_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    for source_path in sorted(source_dir.glob("page_*.png")):
        reconstructed_path = reconstructed_dir / source_path.name
        if not reconstructed_path.is_file():
            continue

        source_image = Image.open(source_path).convert("RGB")
        reconstructed_image = Image.open(reconstructed_path).convert("RGB")
        target_height = max(source_image.height, reconstructed_image.height)
        source_image = resize_to_height(source_image, target_height)
        reconstructed_image = resize_to_height(reconstructed_image, target_height)

        comparison = Image.new(
            "RGB",
            (source_image.width + reconstructed_image.width, target_height),
            "white",
        )
        comparison.paste(source_image, (0, 0))
        comparison.paste(reconstructed_image, (source_image.width, 0))
        comparison.save(output_dir / source_path.name)
        print(f"Comparison {source_path.stem}", flush=True)


def resize_to_height(image, target_height):
    if image.height == target_height:
        return image

    target_width = round(image.width * (target_height / image.height))
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def load_font_policy(font_family):
    if font_family != "times":
        raise ValueError(f"Unsupported font family: {font_family}")

    font_files = {
        "regular": FONT_DIR / "times.ttf",
        "bold": FONT_DIR / "timesbd.ttf",
        "italic": FONT_DIR / "timesi.ttf",
        "bold_italic": FONT_DIR / "timesbi.ttf",
    }

    if not all(path.is_file() for path in font_files.values()):
        return built_in_font_policy()

    styles = {}
    for style, path in font_files.items():
        styles[style] = {
            "style": style,
            "name": f"bookscribe_{style}",
            "fontfile": path,
            "measure": fitz.Font(fontfile=str(path)),
        }

    return {"family": "times", "styles": styles}


def built_in_font_policy():
    styles = {
        "regular": {"style": "regular", "name": "tiro", "fontfile": Path(""), "measure": fitz.Font("tiro")},
        "bold": {"style": "bold", "name": "tibo", "fontfile": Path(""), "measure": fitz.Font("tibo")},
        "italic": {"style": "italic", "name": "tiit", "fontfile": Path(""), "measure": fitz.Font("tiit")},
        "bold_italic": {"style": "bold_italic", "name": "tibi", "fontfile": Path(""), "measure": fitz.Font("tibi")},
    }
    return {"family": "built-in-times", "styles": styles}


def repair_text(text):
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

    return normalize_pdf_text(text)


def normalize_pdf_text(text):
    return (
        text.replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def parse_pages(value):
    pages = []

    for part in value.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            start, end = [int(piece.strip()) for piece in part.split("-", 1)]
            if end < start:
                raise ValueError(f"Invalid page range: {part}")
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))

    return sorted(dict.fromkeys(pages))


def format_page_list(page_numbers):
    if not page_numbers:
        return ""
    if len(page_numbers) > 6:
        return f"{page_numbers[0]}-{page_numbers[-1]} ({len(page_numbers)} pages)"
    return ", ".join(str(page_number) for page_number in page_numbers)


def batches(items, batch_size):
    if batch_size < 1:
        raise ValueError("OCR batch size must be at least 1.")

    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
