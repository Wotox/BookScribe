import contextlib
import tempfile
from pathlib import Path

from PIL import Image


DEFAULT_BACKEND = "unlimited-ocr"
OCR_BACKENDS = ("unlimited-ocr", "easyocr")
UNLIMITED_OCR_MODEL = "baidu/Unlimited-OCR"
UNLIMITED_OCR_PROMPT = "<image>document parsing."
UNLIMITED_OCR_MULTI_PROMPT = "<image>Multi page parsing."
DEFAULT_BATCH_SIZE = 4


class OCRReader:
    def __init__(self, backend=DEFAULT_BACKEND, languages=None, gpu=True):
        if backend not in OCR_BACKENDS:
            backend_list = ", ".join(OCR_BACKENDS)
            raise ValueError(f"Unknown OCR backend: {backend}. Expected one of: {backend_list}")

        self.backend = backend
        self.languages = languages or ["en"]
        self.gpu = gpu
        self._easyocr_reader = None
        self._unlimited_ocr_tokenizer = None
        self._unlimited_ocr_model = None

    def read_text_from_page(self, image):
        if self.backend == "easyocr":
            return self._read_with_easyocr(image)

        return self._read_with_unlimited_ocr(image)

    def read_text_from_pages(self, images):
        if not images:
            return []

        if self.backend == "easyocr" or len(images) == 1:
            return [self.read_text_from_page(image) for image in images]

        return self._read_pages_with_unlimited_ocr(images)

    def read_text_from_image_paths(self, image_paths):
        if not image_paths:
            return []

        if self.backend != "unlimited-ocr":
            raise RuntimeError("Image-path OCR batches are only supported for Unlimited-OCR.")

        if len(image_paths) == 1:
            return [self._read_image_path_with_unlimited_ocr(image_paths[0])]

        return self._read_image_paths_with_unlimited_ocr(image_paths)

    def _read_with_easyocr(self, image):
        results = self._get_easyocr_reader().readtext(image, detail=0, paragraph=True)
        lines = [text.strip() for text in results if text.strip()]
        return "\n\n".join(lines)

    def _read_with_unlimited_ocr(self, image):
        tokenizer, model = self._get_unlimited_ocr_model()

        with tempfile.TemporaryDirectory(prefix="bookscribe_unlimited_ocr_") as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "page.png"
            output_path = temp_path / "output"

            _image_to_pil(image).save(image_path)

            text = self._run_unlimited_ocr_single(tokenizer, model, image_path, output_path)

        return _clean_text(text)

    def _read_pages_with_unlimited_ocr(self, images):
        tokenizer, model = self._get_unlimited_ocr_model()

        with tempfile.TemporaryDirectory(prefix="bookscribe_unlimited_ocr_") as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "output"
            image_paths = []

            for page_index, image in enumerate(images, start=1):
                image_path = temp_path / f"page_{page_index:04}.png"
                _image_to_pil(image).save(image_path)
                image_paths.append(str(image_path))

            text = self._run_unlimited_ocr_multi(tokenizer, model, image_paths, output_path)

        return _split_pages(text, len(images))

    def _read_image_path_with_unlimited_ocr(self, image_path):
        tokenizer, model = self._get_unlimited_ocr_model()

        with tempfile.TemporaryDirectory(prefix="bookscribe_unlimited_ocr_") as temp_dir:
            output_path = Path(temp_dir) / "output"
            text = self._run_unlimited_ocr_single(tokenizer, model, image_path, output_path)

        return _clean_text(text)

    def _read_image_paths_with_unlimited_ocr(self, image_paths):
        tokenizer, model = self._get_unlimited_ocr_model()

        with tempfile.TemporaryDirectory(prefix="bookscribe_unlimited_ocr_") as temp_dir:
            output_path = Path(temp_dir) / "output"
            text = self._run_unlimited_ocr_multi(tokenizer, model, image_paths, output_path)

        return _split_pages(text, len(image_paths))

    def _run_unlimited_ocr_single(self, tokenizer, model, image_path, output_path):
        with _quiet_model_output():
            return model.infer(
                tokenizer,
                prompt=UNLIMITED_OCR_PROMPT,
                image_file=str(image_path),
                output_path=str(output_path),
                base_size=1024,
                image_size=640,
                crop_mode=True,
                max_length=32768,
                no_repeat_ngram_size=35,
                ngram_window=128,
                save_results=False,
                eval_mode=True,
            )

    def _run_unlimited_ocr_multi(self, tokenizer, model, image_paths, output_path):
        with _quiet_model_output():
            return model.infer_multi(
                tokenizer,
                prompt=UNLIMITED_OCR_MULTI_PROMPT,
                image_files=[str(path) for path in image_paths],
                output_path=str(output_path),
                image_size=1024,
                max_length=32768,
                no_repeat_ngram_size=35,
                ngram_window=1024,
                save_results=False,
            )

    def _get_easyocr_reader(self):
        if self._easyocr_reader is None:
            import easyocr

            self._easyocr_reader = easyocr.Reader(self.languages, gpu=self.gpu)

        return self._easyocr_reader

    def _get_unlimited_ocr_model(self):
        if self._unlimited_ocr_model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            if not self.gpu or not torch.cuda.is_available():
                raise RuntimeError(
                    "Unlimited-OCR requires CUDA in its current Hugging Face implementation. "
                    "Install CUDA-enabled PyTorch or run with --ocr-backend easyocr."
                )

            # Unlimited-OCR ships custom Hugging Face model code.
            tokenizer = AutoTokenizer.from_pretrained(
                UNLIMITED_OCR_MODEL,
                trust_remote_code=True,
            )
            model = AutoModel.from_pretrained(
                UNLIMITED_OCR_MODEL,
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=torch.bfloat16,
            )
            self._unlimited_ocr_tokenizer = tokenizer
            self._unlimited_ocr_model = model.eval().cuda()

        return self._unlimited_ocr_tokenizer, self._unlimited_ocr_model


def _image_to_pil(image):
    if image.ndim == 2:
        return Image.fromarray(image)

    if image.ndim == 3 and image.shape[2] == 3:
        return Image.fromarray(image).convert("RGB")

    if image.ndim == 3 and image.shape[2] == 4:
        return Image.fromarray(image).convert("RGB")

    raise ValueError(f"Unsupported page image shape: {image.shape}")


def save_page_image(image, image_path):
    _image_to_pil(image).save(image_path)


def _clean_text(text):
    stop_marker = "<\uff5cend\u2581of\u2581sentence\uff5c>"

    if isinstance(text, tuple):
        text = text[0]

    text = str(text)
    if text.endswith(stop_marker):
        text = text[: -len(stop_marker)]

    return text.strip()


def _quiet_model_output():
    return contextlib.redirect_stdout(_NullOutput())


class _NullOutput:
    encoding = "utf-8"

    def write(self, value):
        return len(value)

    def flush(self):
        pass


def _split_pages(text, expected_count):
    text = _clean_text(text)

    if "<PAGE>" not in text:
        return [text] + [""] * (expected_count - 1)

    pages = [page.strip() for page in text.split("<PAGE>")[1:]]

    if len(pages) < expected_count:
        pages.extend([""] * (expected_count - len(pages)))

    if len(pages) > expected_count:
        pages = pages[: expected_count - 1] + ["\n\n".join(pages[expected_count - 1 :])]

    return pages
