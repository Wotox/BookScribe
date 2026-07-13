import io
from itertools import combinations
import math
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
ALIGNMENT_FONT = fitz.Font("tiro")


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
        if _should_preserve_source_page(source_image, blocks):
            page.insert_image(page.rect, stream=_image_bytes(source_image))
            _insert_invisible_text_blocks(page, blocks)
            return

        entries = []
        for block in blocks:
            rect = _scale_bbox(block["bbox"], page.rect)
            if rect.is_empty or rect.width <= 1 or rect.height <= 1:
                continue

            if block["kind"] == "image":
                crop = _crop_source_region(source_image, block["bbox"])
                entries.append({"kind": "image", "rect": rect, "crop": crop})
                continue

            text = block["text"].strip()
            if not text:
                continue

            crop = _crop_source_region(source_image, block["bbox"])
            layout = _source_text_layout(crop, text)
            entries.append(
                {
                    "kind": block["kind"],
                    "rect": rect,
                    "crop": crop,
                    "text": text,
                    "layout": layout,
                }
            )

        text_entries = [entry for entry in entries if entry["kind"] != "image"]
        word_groups = [
            [word["crop"] for line in entry["layout"] for word in line["words"]]
            if entry["layout"] is not None
            else []
            for entry in text_entries
        ]
        choices = font_classifier.classify_word_groups(
            word_groups,
            [entry["kind"] for entry in text_entries],
        )
        for entry, entry_choices in zip(text_entries, choices):
            if entry["layout"] is None:
                continue

            choice_index = 0
            for line in entry["layout"]:
                for word in line["words"]:
                    word["choice"] = entry_choices[choice_index]
                    choice_index += 1

        body_font_size = _page_body_font_size(text_entries, font_classifier)

        for entry in entries:
            if entry["kind"] == "image":
                page.insert_image(entry["rect"], stream=_image_bytes(entry["crop"]))
            elif entry["layout"] is None:
                _insert_text_fallback(
                    page,
                    entry["rect"],
                    entry["text"],
                    entry["crop"],
                    entry["kind"],
                    font_classifier,
                )
            else:
                _insert_source_layout(
                    page,
                    entry["rect"],
                    entry["crop"],
                    entry["layout"],
                    entry["kind"],
                    font_classifier,
                    body_font_size,
                )


def _should_preserve_source_page(source_image, blocks):
    word_count = sum(
        len(block["text"].split())
        for block in blocks
        if block["kind"] != "image"
    )

    array = np.asarray(source_image, dtype=np.int16)
    channel_range = array.max(axis=2) - array.min(axis=2)
    colorful_fraction = float((channel_range > 25).mean())
    if word_count <= 20 and colorful_fraction > 0.2:
        return True

    return False


def _insert_invisible_text_blocks(page, blocks):
    for block in blocks:
        text = block["text"].strip()
        if block["kind"] == "image" or not text:
            continue

        rect = _scale_bbox(block["bbox"], page.rect)
        font_size = max(4.0, min(12.0, rect.height * 0.45))
        page.insert_textbox(
            rect,
            text,
            fontsize=font_size,
            fontname="helv",
            render_mode=3,
        )


def _source_text_layout(crop, text):
    tokens = text.split()
    if not tokens:
        return []

    ink, lines = _source_line_regions(crop)
    if not lines:
        return None

    target_counts = _target_line_word_counts(lines, tokens)
    if target_counts is None:
        return None

    token_index = 0
    for line, target_count in zip(lines, target_counts):
        line_tokens = tokens[token_index : token_index + target_count]
        token_index += target_count
        aligned = _split_attached_residuals(
            line["initial_words"],
            line["glyph_bands"],
            line["bottom"] - line["top"],
            line_tokens,
        )
        if aligned is None:
            aligned = _align_initial_word_groups(
                line["initial_words"],
                line["glyph_bands"],
                line["bottom"] - line["top"],
                line_tokens,
            )
        if aligned is None:
            word_bands = _word_bands_for_tokens(line["glyph_bands"], line_tokens)
            residual_bands = []
        else:
            word_bands, residual_bands = aligned
        if len(word_bands) != len(line_tokens):
            return None

        line["left"] = word_bands[0][0]
        line["right"] = word_bands[-1][1]
        line["residual_bands"] = residual_bands
        line["words"] = []
        for token, (left, right) in zip(line_tokens, word_bands):
            padding = 2
            word_crop = crop.crop(
                (
                    max(0, left - padding),
                    max(0, line["top"] - padding),
                    min(crop.width, right + padding),
                    min(crop.height, line["bottom"] + padding),
                )
            )
            line["words"].append(
                {
                    "text": token,
                    "left": left,
                    "right": right,
                    "crop": word_crop,
                }
            )

    if token_index != len(tokens):
        return None

    return lines


def _source_line_regions(crop):
    gray = ImageOps.autocontrast(ImageOps.grayscale(crop))
    array = np.asarray(gray, dtype=np.uint8)
    threshold = min(235, max(35, int(array.mean() - array.std() * 0.15)))
    ink = array < threshold
    row_counts = ink.sum(axis=1)
    if not len(row_counts) or int(row_counts.max()) <= 0:
        return ink, []

    active = row_counts > max(2, row_counts.max() * 0.08)
    bands = _merge_line_fragments(_axis_bands(active), max_gap=3)
    lines = []

    for top, bottom in bands:
        if bottom - top < 2:
            continue

        columns = ink[top:bottom].sum(axis=0) > 0
        glyph_bands = _axis_bands(columns)
        if not glyph_bands:
            continue

        word_gap = max(3, int(round((bottom - top) * 0.18)))
        initial_words = _merge_close_bands(glyph_bands, word_gap)
        lines.append(
            {
                "top": top,
                "bottom": bottom,
                "glyph_bands": glyph_bands,
                "initial_words": initial_words,
                "initial_count": len(initial_words),
            }
        )

    return ink, lines


def _align_initial_word_groups(groups, glyph_bands, line_height, tokens):
    if len(groups) <= len(tokens) or len(groups) - len(tokens) > 4:
        return None
    if len(tokens) > 20:
        return None

    expected_widths = [
        max(ALIGNMENT_FONT.text_length(token, fontsize=1.0), 1e-6)
        for token in tokens
    ]
    best = None
    for selected_indices in combinations(range(len(groups)), len(tokens)):
        ratios = np.asarray(
            [
                (groups[group_index][1] - groups[group_index][0]) / expected_width
                for group_index, expected_width in zip(
                    selected_indices,
                    expected_widths,
                )
            ],
            dtype=np.float32,
        )
        cost = float(np.log(ratios).std())
        if best is None or cost < best[0]:
            best = (cost, selected_indices)

    if best is None or best[0] > 0.45:
        return None

    selected = set(best[1])
    word_bands = [groups[index] for index in best[1]]
    residual_bands = [
        group
        for index, group in enumerate(groups)
        if index not in selected
    ]
    if not all(
        _is_visual_residual(group, glyph_bands, line_height)
        for group in residual_bands
    ):
        return None
    return word_bands, residual_bands


def _split_attached_residuals(groups, glyph_bands, line_height, tokens):
    if len(groups) != len(tokens) or len(tokens) < 2:
        return None

    expected_widths = np.asarray(
        [
            max(ALIGNMENT_FONT.text_length(token, fontsize=1.0), 1e-6)
            for token in tokens
        ],
        dtype=np.float32,
    )
    observed_widths = np.asarray(
        [right - left for left, right in groups],
        dtype=np.float32,
    )
    ratios = observed_widths / expected_widths
    median_ratio = float(np.median(ratios))
    outliers = np.nonzero(ratios > median_ratio * 2.2)[0]
    if len(outliers) != 1:
        return None

    index = int(outliers[0])
    left, right = groups[index]
    expected_end = left + expected_widths[index] * median_ratio
    contained = [
        band
        for band in glyph_bands
        if left <= band[0] and band[1] <= right
    ]
    if len(contained) < 2:
        return None

    split_index = min(
        range(len(contained) - 1),
        key=lambda candidate: abs(contained[candidate][1] - expected_end),
    )
    word_band = (left, contained[split_index][1])
    residual_band = (contained[split_index + 1][0], right)
    if residual_band[1] - residual_band[0] <= word_band[1] - word_band[0]:
        return None
    if not _is_visual_residual(residual_band, glyph_bands, line_height):
        return None

    word_bands = list(groups)
    word_bands[index] = word_band
    return word_bands, [residual_band]


def _is_visual_residual(region, glyph_bands, line_height):
    left, right = region
    contained = [
        band
        for band in glyph_bands
        if left <= band[0] and band[1] <= right
    ]
    region_width = right - left
    if len(contained) == 1:
        return region_width >= line_height * 8
    if len(contained) < 5 or region_width < line_height * 4:
        return False

    widths = np.asarray(
        [band[1] - band[0] for band in contained],
        dtype=np.float32,
    )
    gaps = np.asarray(
        [
            contained[index + 1][0] - contained[index][1]
            for index in range(len(contained) - 1)
        ],
        dtype=np.float32,
    )
    width_cv = float(widths.std() / max(widths.mean(), 1e-6))
    gap_cv = float(gaps.std() / max(gaps.mean(), 1e-6))
    return width_cv <= 0.45 and gap_cv <= 0.8


def _target_line_word_counts(lines, tokens):
    counts = [line["initial_count"] for line in lines]
    difference = len(tokens) - sum(counts)

    while difference > 0:
        candidates = []
        for index, (line, count) in enumerate(zip(lines, counts)):
            gaps = _sorted_gaps(line["glyph_bands"])
            next_gap = gaps[count - 1][0] if count - 1 < len(gaps) else -1
            candidates.append((next_gap, -index, index))

        next_gap, _negative_index, index = max(candidates)
        if next_gap < 0:
            return None
        counts[index] += 1
        difference -= 1

    while difference < 0:
        candidates = []
        for index, (line, count) in enumerate(zip(lines, counts)):
            if count <= 1:
                continue
            gaps = _sorted_gaps(line["glyph_bands"])
            weakest_gap = gaps[count - 2][0]
            candidates.append((weakest_gap, index))

        if not candidates:
            return None
        _weakest_gap, index = min(candidates)
        counts[index] -= 1
        difference += 1

    return _optimize_line_word_counts(lines, tokens, counts)


def _optimize_line_word_counts(lines, tokens, counts):
    counts = list(counts)
    current_cost = _line_alignment_cost(lines, tokens, counts)
    ratios = _line_alignment_ratios(lines, tokens, counts)
    target_ratio = float(np.median(ratios))
    candidate = _line_counts_for_scale(lines, tokens, target_ratio)
    candidate_cost = _line_alignment_cost(lines, tokens, candidate)
    return candidate if current_cost - candidate_cost >= 0.01 else counts


def _line_counts_for_scale(lines, tokens, target_ratio):
    state = {(0, 0): (0.0, [])}
    line_count = len(lines)

    for line_index, line in enumerate(lines):
        next_state = {}
        initial_count = line["initial_count"]
        observed_width = line["glyph_bands"][-1][1] - line["glyph_bands"][0][0]

        for (_processed_lines, token_index), (cost, counts) in state.items():
            remaining_lines = line_count - line_index - 1
            maximum_count = min(
                len(line["glyph_bands"]),
                initial_count + 5,
                len(tokens) - token_index - remaining_lines,
            )
            for count in range(max(1, initial_count - 5), maximum_count + 1):
                text = " ".join(tokens[token_index : token_index + count])
                measured_width = ALIGNMENT_FONT.text_length(text, fontsize=1.0)
                ratio = observed_width / max(measured_width, 1e-6)
                line_cost = math.log(ratio / target_ratio) ** 2
                line_cost += 0.0005 * (count - initial_count) ** 2
                key = (line_index + 1, token_index + count)
                candidate = (cost + line_cost, counts + [count])
                if key not in next_state or candidate[0] < next_state[key][0]:
                    next_state[key] = candidate

        state = next_state

    result = state.get((line_count, len(tokens)))
    return result[1] if result is not None else [line["initial_count"] for line in lines]


def _line_alignment_cost(lines, tokens, counts):
    ratios = _line_alignment_ratios(lines, tokens, counts)
    return float(ratios.std() / max(ratios.mean(), 1e-6))


def _line_alignment_ratios(lines, tokens, counts):
    ratios = []
    token_index = 0

    for line, count in zip(lines, counts):
        line_text = " ".join(tokens[token_index : token_index + count])
        token_index += count
        measured_width = ALIGNMENT_FONT.text_length(line_text, fontsize=1.0)
        observed_width = line["glyph_bands"][-1][1] - line["glyph_bands"][0][0]
        ratios.append(observed_width / max(measured_width, 1e-6))

    return np.asarray(ratios, dtype=np.float32)


def _word_bands_for_tokens(glyph_bands, tokens):
    word_count = len(tokens)
    if word_count < 1 or word_count > len(glyph_bands):
        return []

    line_span = glyph_bands[-1][1] - glyph_bands[0][0]
    measured_line = ALIGNMENT_FONT.text_length(" ".join(tokens), fontsize=1.0)
    scale = line_span / max(measured_line, 1e-6)
    state = {(0, 0): (0.0, [])}

    for token_index, token in enumerate(tokens):
        next_state = {}
        remaining_tokens = word_count - token_index - 1
        expected_width = max(
            ALIGNMENT_FONT.text_length(token, fontsize=1.0) * scale,
            1.0,
        )

        for (_processed_tokens, start), (cost, words) in state.items():
            maximum_end = len(glyph_bands) - remaining_tokens
            for end in range(start + 1, maximum_end + 1):
                observed_width = glyph_bands[end - 1][1] - glyph_bands[start][0]
                candidate_cost = cost + math.log(observed_width / expected_width) ** 2
                if end < len(glyph_bands):
                    gap = glyph_bands[end][0] - glyph_bands[end - 1][1]
                    candidate_cost -= min(gap, 12) * 0.015

                key = (token_index + 1, end)
                candidate = (
                    candidate_cost,
                    words + [(glyph_bands[start][0], glyph_bands[end - 1][1])],
                )
                if key not in next_state or candidate_cost < next_state[key][0]:
                    next_state[key] = candidate

        state = next_state

    result = state.get((word_count, len(glyph_bands)))
    return result[1] if result is not None else []


def _sorted_gaps(bands):
    return sorted(
        [
            (bands[index + 1][0] - bands[index][1], index)
            for index in range(len(bands) - 1)
        ],
        reverse=True,
    )


def _axis_bands(active):
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


def _merge_close_bands(bands, max_gap):
    merged = []
    for band in bands:
        if merged and band[0] - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], band[1])
        else:
            merged.append(band)
    return merged


def _merge_line_fragments(bands, max_gap):
    merged = []
    for band in bands:
        previous_height = merged[-1][1] - merged[-1][0] if merged else 0
        band_height = band[1] - band[0]
        is_fragment = min(previous_height, band_height) <= 3
        if merged and band[0] - merged[-1][1] <= max_gap and is_fragment:
            merged[-1] = (merged[-1][0], band[1])
        else:
            merged.append(band)
    return merged


def _page_body_font_size(text_entries, font_classifier):
    sizes = []
    for entry in text_entries:
        if entry["kind"] != "text" or entry["layout"] is None:
            continue

        x_scale = entry["rect"].width / entry["crop"].width
        for line in entry["layout"]:
            for word in line["words"]:
                measured_width = font_classifier.measure_width(
                    word["text"],
                    word["choice"],
                    1.0,
                )
                source_width = (word["right"] - word["left"]) * x_scale
                sizes.append(source_width / max(measured_width, 1e-6))

    if not sizes:
        return None

    return max(4.0, min(12.0, float(np.median(sizes))))


def _line_width_at_one(words, font_classifier):
    width = 0.0
    for index, word in enumerate(words):
        choice = word["choice"]
        width += font_classifier.measure_width(word["text"], choice, 1.0)
        if index < len(words) - 1:
            width += font_classifier.measure_width(" ", choice, 1.0)
    return width


def _insert_source_layout(
    page,
    rect,
    crop,
    layout,
    kind,
    font_classifier,
    body_font_size,
):
    x_scale = rect.width / crop.width
    y_scale = rect.height / crop.height
    residual_image = _layout_residual_image(crop, layout)
    if residual_image is not None:
        page.insert_image(rect, stream=_image_bytes(residual_image), overlay=True)

    median_line_height = float(
        np.median([line["bottom"] - line["top"] for line in layout])
    )

    for line in layout:
        words = line["words"]
        target_width = (line["right"] - line["left"]) * x_scale
        width_at_one = _line_width_at_one(words, font_classifier)

        if kind == "text" and body_font_size is not None:
            font_size = body_font_size
        else:
            height_size = median_line_height * y_scale * 1.5
            width_size = target_width / max(width_at_one, 1e-6)
            font_size = max(4.0, min(24.0, height_size, width_size))
        first_choice = words[0]["choice"]
        baseline = (
            rect.y0
            + line["bottom"] * y_scale
            + first_choice.measure_font.descender * font_size
        )
        text_widths, natural_spaces = _word_and_space_widths(
            words,
            font_size,
            font_classifier,
        )
        natural_width = sum(text_widths) + sum(natural_spaces)
        if natural_width > target_width:
            font_size *= target_width / natural_width
            text_widths, natural_spaces = _word_and_space_widths(
                words,
                font_size,
                font_classifier,
            )

        if natural_spaces:
            extra_space = max(
                0.0,
                target_width - sum(text_widths) - sum(natural_spaces),
            ) / len(natural_spaces)
            spaces = [width + extra_space for width in natural_spaces]
        else:
            spaces = []

        use_source_positions = bool(line.get("residual_bands"))
        x = rect.x0 + line["left"] * x_scale
        for index, (word, text_width) in enumerate(zip(words, text_widths)):
            choice = word["choice"]
            visible_text = word["text"]
            if index < len(words) - 1:
                visible_text += " "
            if use_source_positions:
                x = rect.x0 + word["left"] * x_scale
            page.insert_text(
                (x, baseline),
                visible_text,
                fontsize=font_size,
                **font_classifier.insert_kwargs(choice),
                color=(0, 0, 0),
            )
            x += text_width
            if not use_source_positions and index < len(spaces):
                x += spaces[index]


def _layout_residual_image(crop, layout):
    residual_regions = [
        (line["top"], line["bottom"], left, right)
        for line in layout
        for left, right in line.get("residual_bands", [])
    ]
    if not residual_regions:
        return None

    gray = np.asarray(ImageOps.grayscale(crop), dtype=np.uint8)
    alpha = np.zeros_like(gray, dtype=np.uint8)
    source_alpha = np.where(gray < 180, 255 - gray, 0).astype(np.uint8)
    for top, bottom, left, right in residual_regions:
        padding = 2
        y0 = max(0, top - padding)
        y1 = min(crop.height, bottom + padding)
        x0 = max(0, left - padding)
        x1 = min(crop.width, right + padding)
        alpha[y0:y1, x0:x1] = source_alpha[y0:y1, x0:x1]

    rgba = np.zeros((crop.height, crop.width, 4), dtype=np.uint8)
    rgba[:, :, 3] = alpha
    return Image.fromarray(rgba, mode="RGBA")


def _word_and_space_widths(words, font_size, font_classifier):
    text_widths = []
    spaces = []
    for index, word in enumerate(words):
        choice = word["choice"]
        text_widths.append(
            font_classifier.measure_width(word["text"], choice, font_size)
        )
        if index < len(words) - 1:
            spaces.append(font_classifier.measure_width(" ", choice, font_size))
    return text_widths, spaces


def _insert_text_fallback(page, rect, text, crop, kind, font_classifier):
    if kind == "title":
        base_choice = font_classifier.choices["bold"]
    else:
        base_choice = font_classifier.classify(crop)
    font_size, lines = _fit_lines(text, rect, base_choice, kind, font_classifier)
    y = rect.y0 + font_size

    for line in lines:
        page.insert_text(
            (rect.x0, y),
            line,
            fontsize=font_size,
            **font_classifier.insert_kwargs(base_choice),
            color=(0, 0, 0),
        )
        y += _line_height(font_size)


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
