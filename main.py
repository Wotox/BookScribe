from pathlib import Path
import argparse
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, ProcessPoolExecutor, wait
import sys
import tempfile

from ocr import DEFAULT_BACKEND, DEFAULT_BATCH_SIZE, OCR_BACKENDS, OCRReader, save_page_image
from pdf_opener import render_pages
from pdf_writer import write_text_pdf, write_unlimited_ocr_pdf


DEFAULT_UNLIMITED_OCR_WORKERS = 2
DEFAULT_OTHER_WORKERS = 1
_WORKER_OCR_READER = None


def output_path_for(input_path, page_numbers=None):
    if page_numbers:
        suffix = _page_selection_suffix(page_numbers)
        return input_path.with_name(f"{input_path.stem}_{suffix}_text.pdf")

    return input_path.with_name(f"{input_path.stem}_text.pdf")


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="BookScribe")
    parser.add_argument("pdf_path", help="Path to the scanned PDF to convert.")
    parser.add_argument(
        "--ocr-backend",
        choices=OCR_BACKENDS,
        default=DEFAULT_BACKEND,
        help=f"OCR backend to use. Default: {DEFAULT_BACKEND}.",
    )
    parser.add_argument(
        "--ocr-batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Pages to OCR per model call. Default: {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--ocr-workers",
        type=int,
        default=None,
        help="Parallel OCR worker processes. Default: 2 for Unlimited-OCR, 1 for EasyOCR.",
    )
    parser.add_argument(
        "--pages",
        default=None,
        help="Pages to process, for example 1-20 or 1,3,5. Default: all pages.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.ocr_batch_size < 1:
        print("BookScribe")
        print("OCR batch size must be at least 1.")
        return 2

    ocr_workers = (
        _default_workers_for(args.ocr_backend)
        if args.ocr_workers is None
        else args.ocr_workers
    )
    if ocr_workers < 1:
        print("BookScribe")
        print("OCR workers must be at least 1.")
        return 2

    if ocr_workers > 1 and args.ocr_backend != "unlimited-ocr":
        print("BookScribe")
        print("Parallel OCR workers are only supported for Unlimited-OCR.")
        return 2

    try:
        page_numbers = _parse_page_selection(args.pages)
    except ValueError as exc:
        print("BookScribe")
        print(exc)
        return 2

    input_path = Path(args.pdf_path).expanduser()
    if not input_path.is_file():
        print("BookScribe")
        print(f"Input file not found: {input_path}")
        return 1

    print("BookScribe")
    print(f"OCR backend: {args.ocr_backend}")
    print(f"OCR batch size: {args.ocr_batch_size}")
    print(f"OCR workers: {ocr_workers}")
    if page_numbers:
        print(f"Pages: {_format_page_selection(page_numbers)}")
    output_path = output_path_for(input_path, page_numbers)

    if args.ocr_backend == "unlimited-ocr":
        _write_unlimited_ocr_pdf(
            input_path,
            output_path,
            args.ocr_batch_size,
            ocr_workers,
            page_numbers,
        )
    else:
        page_texts = _read_pages(input_path, args.ocr_backend, args.ocr_batch_size, page_numbers)
        write_text_pdf(page_texts, output_path)

    print(f"Created: {output_path}")
    return 0


def _default_workers_for(ocr_backend):
    if ocr_backend == "unlimited-ocr":
        return DEFAULT_UNLIMITED_OCR_WORKERS

    return DEFAULT_OTHER_WORKERS


def _read_pages(input_path, ocr_backend, batch_size, page_numbers=None):
    page_texts = []
    ocr_reader = OCRReader(backend=ocr_backend)

    pages = render_pages(input_path, page_numbers=page_numbers)
    for page_batch in _page_batches(pages, batch_size):
        batch_page_numbers = [page_number for page_number, _image in page_batch]
        images = [image for _page_number, image in page_batch]
        _print_ocr_batch(batch_page_numbers)
        texts = ocr_reader.read_text_from_pages(images)
        page_texts.extend(texts)
        _print_processed_pages(batch_page_numbers)

    return page_texts


def _write_unlimited_ocr_pdf(input_path, output_path, batch_size, ocr_workers, page_numbers=None):
    with tempfile.TemporaryDirectory(prefix="bookscribe_pages_") as temp_dir:
        page_results = _render_page_images(input_path, Path(temp_dir), page_numbers)

        if ocr_workers == 1:
            page_texts = _ocr_page_images(page_results, batch_size)
        else:
            page_texts = _ocr_page_images_parallel(page_results, batch_size, ocr_workers)

        for page_result in page_results:
            page_result["text"] = page_texts[page_result["page_number"]]

        write_unlimited_ocr_pdf(input_path, page_results, output_path)


def _render_page_images(input_path, temp_path, page_numbers=None):
    page_results = []

    for page_number, image in render_pages(input_path, page_numbers=page_numbers):
        image_path = temp_path / f"page_{page_number:05}.png"
        save_page_image(image, image_path)
        page_results.append(
            {
                "page_number": page_number,
                "image_path": str(image_path),
            }
        )

    return page_results


def _ocr_page_images(page_results, batch_size):
    ocr_reader = OCRReader(backend="unlimited-ocr")
    page_texts = {}

    for page_batch in _page_batches(page_results, batch_size):
        page_numbers = [page["page_number"] for page in page_batch]
        image_paths = [page["image_path"] for page in page_batch]
        _print_ocr_batch(page_numbers)
        texts = ocr_reader.read_text_from_image_paths(image_paths)
        _store_page_texts(page_texts, page_numbers, texts)
        _print_processed_pages(page_numbers)

    return page_texts


def _ocr_page_images_parallel(page_results, batch_size, ocr_workers):
    page_texts = {}
    pending = {}
    max_pending = ocr_workers * 2

    with ProcessPoolExecutor(
        max_workers=ocr_workers,
        initializer=_init_ocr_worker,
        initargs=("unlimited-ocr",),
    ) as executor:
        for page_batch in _page_batches(page_results, batch_size):
            page_numbers = [page["page_number"] for page in page_batch]
            image_paths = [page["image_path"] for page in page_batch]
            _print_ocr_batch(page_numbers, prefix="Queue")
            future = executor.submit(_ocr_image_paths_worker, image_paths)
            pending[future] = page_numbers

            if len(pending) >= max_pending:
                _collect_finished_ocr_batches(pending, page_texts, FIRST_COMPLETED)

        _collect_finished_ocr_batches(pending, page_texts, ALL_COMPLETED)

    return page_texts


def _collect_finished_ocr_batches(pending, page_texts, return_when):
    if not pending:
        return

    done, _not_done = wait(pending, return_when=return_when)

    for future in done:
        page_numbers = pending.pop(future)
        texts = future.result()
        _store_page_texts(page_texts, page_numbers, texts)
        for page_number in page_numbers:
            _print_processed_page(page_number)


def _store_page_texts(page_texts, page_numbers, texts):
    if len(texts) != len(page_numbers):
        raise RuntimeError(
            f"OCR worker returned {len(texts)} pages for {len(page_numbers)} inputs."
        )

    for page_number, text in zip(page_numbers, texts):
        page_texts[page_number] = text


def _init_ocr_worker(ocr_backend):
    global _WORKER_OCR_READER
    _WORKER_OCR_READER = OCRReader(backend=ocr_backend)


def _ocr_image_paths_worker(image_paths):
    return _WORKER_OCR_READER.read_text_from_image_paths(image_paths)


def _print_ocr_batch(page_numbers, prefix="OCR"):
    if len(page_numbers) == 1:
        print(f"{prefix} page {page_numbers[0]}...", flush=True)
    else:
        print(f"{prefix} pages {page_numbers[0]}-{page_numbers[-1]}...", flush=True)


def _print_processed_pages(page_numbers):
    for page_number in page_numbers:
        _print_processed_page(page_number)


def _print_processed_page(page_number):
    print(f"Processed page {page_number}", flush=True)


def _page_batches(pages, batch_size):
    batch = []

    for page in pages:
        batch.append(page)

        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def _parse_page_selection(value):
    if value is None:
        return None

    page_numbers = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if start < 1 or end < start:
                raise ValueError(f"Invalid page range: {part}")
            page_numbers.extend(range(start, end + 1))
        else:
            page_number = int(part)
            if page_number < 1:
                raise ValueError(f"Invalid page number: {page_number}")
            page_numbers.append(page_number)

    if not page_numbers:
        raise ValueError("No pages selected.")

    return sorted(dict.fromkeys(page_numbers))


def _page_selection_suffix(page_numbers):
    if len(page_numbers) == 1:
        return f"page_{page_numbers[0]:03}"

    return f"pages_{page_numbers[0]:03}_{page_numbers[-1]:03}"


def _format_page_selection(page_numbers):
    if len(page_numbers) == 1:
        return str(page_numbers[0])

    contiguous = page_numbers == list(range(page_numbers[0], page_numbers[-1] + 1))
    if contiguous:
        return f"{page_numbers[0]}-{page_numbers[-1]}"

    return ",".join(str(page_number) for page_number in page_numbers)


if __name__ == "__main__":
    raise SystemExit(main())
