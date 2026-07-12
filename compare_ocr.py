import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import fitz
import numpy as np
from PIL import Image

from ocr import OCRReader


DEFAULT_PDF = Path(
    r"C:\Users\hplus\AI_Dev\bookscribe\Scanned"
    r"\The_Singapore_story_memoirs_of_Lee_Kuan_Yew_Lee,_Kuan_Yew,_Kuan.pdf"
)
DEFAULT_PAGES = [1, 3, 5]
DEFAULT_BACKENDS = [
    "easyocr",
    "unlimited-ocr",
    "surya-ocr",
    "paddleocr-vl",
    "nuextract3",
]
TRANSFORMERS5_BACKENDS = {"paddleocr-vl", "nuextract3"}
TRANSFORMERS5_VENDOR_DIR = Path(".ocr_vendor") / "transformers5"
LLAMA_CPP_BINARY = Path(".ocr_vendor") / "llama.cpp" / "b9843" / "llama-server.exe"
LATEST_RESULTS_PATH = Path("ocr_comparison") / "latest_results.md"
LATEST_SUMMARY_PATH = Path("ocr_comparison") / "latest_summary.json"


def main():
    args = parse_args()

    if args.backend:
        return run_backend(args)

    return run_comparison(args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare OCR backends on selected PDF pages."
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--pages", default="1,3,5")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--backends", default=",".join(DEFAULT_BACKENDS))
    parser.add_argument("--backend", choices=DEFAULT_BACKENDS, default=None)
    parser.add_argument("--page-dir", type=Path, default=None)
    return parser.parse_args()


def run_comparison(args):
    pages = parse_pages(args.pages)
    output_dir = args.output_dir or default_output_dir()
    page_dir = output_dir / "pages"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rendering pages {', '.join(str(page) for page in pages)}")
    rendered_pages = render_selected_pages(args.pdf, pages, page_dir)
    write_json(
        output_dir / "pages.json",
        {"pdf": str(args.pdf), "pages": rendered_pages},
    )

    results = []
    for backend in parse_backends(args.backends):
        backend_dir = output_dir / backend
        backend_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {backend} ===", flush=True)

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--backend",
            backend,
            "--page-dir",
            str(page_dir),
            "--output-dir",
            str(backend_dir),
        ]
        started_at = time.perf_counter()
        completed = subprocess.run(command, env=worker_env(backend))
        elapsed = time.perf_counter() - started_at

        result_path = backend_dir / "result.json"
        if result_path.is_file():
            result = read_json(result_path)
        else:
            result = {
                "backend": backend,
                "status": "failed",
                "error": f"Worker exited with code {completed.returncode} without result.json",
            }

        result["worker_exit_code"] = completed.returncode
        result["elapsed_seconds_total"] = round(elapsed, 3)
        results.append(result)

    summary = {
        "pdf": str(args.pdf),
        "requested_pages": pages,
        "pages": rendered_pages,
        "output_dir": str(output_dir),
        "results": results,
    }
    write_json(output_dir / "summary.json", summary)
    write_markdown_report(output_dir / "results_by_page.md", summary)
    write_json(LATEST_SUMMARY_PATH, summary)
    write_markdown_report(LATEST_RESULTS_PATH, summary)
    print_summary(summary)
    print(f"Page-by-page report: {output_dir / 'results_by_page.md'}")
    print(f"Latest report: {LATEST_RESULTS_PATH}")
    return 0


def run_backend(args):
    if args.page_dir is None or args.output_dir is None:
        print("--backend requires --page-dir and --output-dir")
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    page_paths = sorted(args.page_dir.glob("page_*.png"))
    started_at = time.perf_counter()
    result = {
        "backend": args.backend,
        "status": "ok",
        "pages": [],
        "error": None,
    }

    try:
        reader = make_backend_reader(args.backend)

        for page_path in page_paths:
            page_number = page_number_from_path(page_path)
            page_started_at = time.perf_counter()
            print(f"{args.backend}: page {page_number}", flush=True)
            text = read_page(reader, args.backend, page_path)
            page_elapsed = time.perf_counter() - page_started_at
            output_path = args.output_dir / f"page_{page_number:03}.md"
            output_path.write_text(text, encoding="utf-8")
            result["pages"].append(
                {
                    "page": page_number,
                    "output_file": str(output_path),
                    "char_count": len(text),
                    "elapsed_seconds": round(page_elapsed, 3),
                }
            )
            print(f"{args.backend}: processed page {page_number}", flush=True)
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], flush=True)
    finally:
        result["elapsed_seconds_total"] = round(time.perf_counter() - started_at, 3)
        write_json(args.output_dir / "result.json", result)

    return 0 if result["status"] == "ok" else 1


def make_backend_reader(backend):
    if backend in ("easyocr", "unlimited-ocr"):
        return OCRReader(backend=backend)

    if backend == "surya-ocr":
        from surya.inference import SuryaInferenceManager
        from surya.recognition import RecognitionPredictor

        manager = SuryaInferenceManager()
        return RecognitionPredictor(manager)

    if backend == "paddleocr-vl":
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_id = "PaddlePaddle/PaddleOCR-VL-1.6"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).to(device).eval()
        processor = AutoProcessor.from_pretrained(model_id)
        return {"model": model, "processor": processor, "device": device}

    if backend == "nuextract3":
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_id = "numind/NuExtract3"
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        return {"model": model, "processor": processor}

    raise ValueError(f"Unknown backend: {backend}")


def read_page(reader, backend, page_path):
    if backend == "easyocr":
        image = np.array(Image.open(page_path).convert("RGB"))
        return reader.read_text_from_page(image)

    if backend == "unlimited-ocr":
        return reader.read_text_from_image_paths([str(page_path)])[0]

    if backend == "surya-ocr":
        image = Image.open(page_path).convert("RGB")
        prediction = reader([image])[0]
        return surya_prediction_text(prediction)

    if backend == "paddleocr-vl":
        return run_paddleocr_vl(reader, page_path)

    if backend == "nuextract3":
        return run_nuextract3(reader, page_path)

    raise ValueError(f"Unknown backend: {backend}")


def run_paddleocr_vl(reader, page_path):
    import torch

    image = Image.open(page_path).convert("RGB")
    processor = reader["processor"]
    model = reader["model"]
    prompt = "OCR:"
    max_pixels = 1280 * 28 * 28
    min_pixels = processor.image_processor.size.shortest_edge
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={
            "size": {
                "shortest_edge": min_pixels,
                "longest_edge": max_pixels,
            }
        },
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=4096)

    return processor.decode(outputs[0][inputs["input_ids"].shape[-1] : -1]).strip()


def run_nuextract3(reader, page_path):
    import torch

    image = Image.open(page_path).convert("RGB")
    processor = reader["processor"]
    model = reader["model"]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                }
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        mode="markdown",
        enable_thinking=False,
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=8192,
            do_sample=False,
        )

    generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def surya_prediction_text(prediction):
    data = prediction_to_data(prediction)
    blocks = data.get("blocks") if isinstance(data, dict) else None
    if blocks:
        parts = []
        for block in blocks:
            html = block.get("html") if isinstance(block, dict) else None
            text = block.get("text") if isinstance(block, dict) else None
            if html:
                parts.append(str(html))
            elif text:
                parts.append(str(text))
        return "\n\n".join(parts).strip()

    for field_name in ("markdown", "html", "text"):
        value = getattr(prediction, field_name, None)
        if value:
            return str(value).strip()

    return json.dumps(data, ensure_ascii=False, indent=2)


def prediction_to_data(prediction):
    if isinstance(prediction, dict):
        return prediction

    if hasattr(prediction, "model_dump"):
        return prediction.model_dump()

    if hasattr(prediction, "dict"):
        return prediction.dict()

    if hasattr(prediction, "__dict__"):
        return prediction.__dict__

    return {"repr": repr(prediction)}


def render_selected_pages(pdf_path, pages, output_dir, zoom=2):
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_pages = []
    wanted = set(pages)
    document = fitz.open(str(pdf_path))

    try:
        matrix = fitz.Matrix(zoom, zoom)
        for page_number in pages:
            page = document[page_number - 1]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
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
    finally:
        document.close()

    rendered_numbers = {page["page"] for page in rendered_pages}
    missing = wanted - rendered_numbers
    if missing:
        raise ValueError(f"Pages not rendered: {sorted(missing)}")

    return rendered_pages


def parse_pages(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_backends(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def page_number_from_path(path):
    return int(path.stem.split("_")[-1])


def default_output_dir():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("ocr_comparison") / timestamp


def worker_env(backend):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    if backend in TRANSFORMERS5_BACKENDS:
        vendor_path = str(TRANSFORMERS5_VENDOR_DIR.resolve())
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            env["PYTHONPATH"] = vendor_path + os.pathsep + existing_pythonpath
        else:
            env["PYTHONPATH"] = vendor_path

    if backend == "surya-ocr" and LLAMA_CPP_BINARY.is_file():
        llama_dir = str(LLAMA_CPP_BINARY.resolve().parent)
        env["SURYA_INFERENCE_BACKEND"] = "llamacpp"
        env["LLAMA_CPP_BINARY"] = str(LLAMA_CPP_BINARY.resolve())
        env["PATH"] = llama_dir + os.pathsep + env.get("PATH", "")

    return env


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_markdown_report(path, summary):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    page_numbers = report_page_numbers(summary)

    lines = [
        "# OCR Comparison Results",
        "",
        f"Source PDF: `{summary.get('pdf', '')}`",
        f"Run output: `{summary.get('output_dir', '')}`",
        f"Pages: {', '.join(str(page) for page in page_numbers)}",
        "",
        "## Backend Summary",
        "",
        "| Backend | Status | Total time | Extracted text |",
        "| --- | --- | ---: | --- |",
    ]

    for result in summary.get("results", []):
        pages = result.get("pages") or []
        page_bits = ", ".join(
            f"p{page['page']}: {page.get('char_count', 0)} chars" for page in pages
        )
        if not page_bits and result.get("error"):
            page_bits = "failed before page output"
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_table_cell(result.get("backend", "")),
                    escape_table_cell(result.get("status", "")),
                    escape_table_cell(format_seconds(result.get("elapsed_seconds_total"))),
                    escape_table_cell(page_bits),
                ]
            )
            + " |"
        )

    for page_number in page_numbers:
        lines.extend(["", f"## Page {page_number}", ""])

        for result in summary.get("results", []):
            backend = result.get("backend", "unknown")
            page = find_page_entry(result, page_number)
            lines.append(f"### {backend}")

            if page is None:
                if result.get("status") == "failed":
                    lines.append(f"Status: failed. Error: {result.get('error')}")
                else:
                    lines.append("No output for this page.")
                lines.append("")
                continue

            lines.append(
                "Chars: "
                f"{page.get('char_count', 0)}. "
                f"Time: {format_seconds(page.get('elapsed_seconds'))}. "
                f"File: `{page.get('output_file', '')}`"
            )
            lines.extend(["", "````text", read_report_text(page.get("output_file")), "````", ""])

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def report_page_numbers(summary):
    page_numbers = set(summary.get("requested_pages") or [])
    for result in summary.get("results", []):
        for page in result.get("pages") or []:
            page_numbers.add(page["page"])
    return sorted(page_numbers)


def find_page_entry(result, page_number):
    for page in result.get("pages") or []:
        if page.get("page") == page_number:
            return page
    return None


def read_report_text(path):
    if not path:
        return ""

    try:
        return Path(path).read_text(encoding="utf-8").rstrip()
    except OSError as exc:
        return f"[Could not read extracted text file: {exc}]"


def format_seconds(value):
    if value is None:
        return ""
    return f"{value}s"


def escape_table_cell(value):
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def print_summary(summary):
    print("\nSummary")
    print(f"Output: {summary['output_dir']}")
    for result in summary["results"]:
        status = result.get("status")
        elapsed = result.get("elapsed_seconds_total")
        pages = result.get("pages") or []
        page_bits = ", ".join(
            f"p{page['page']}={page['char_count']} chars" for page in pages
        )
        print(f"- {result['backend']}: {status}, {elapsed}s, {page_bits}")
        if result.get("error"):
            print(f"  error: {result['error']}")


if __name__ == "__main__":
    raise SystemExit(main())
