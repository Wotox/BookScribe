from dataclasses import dataclass
from pathlib import Path
import random

import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
import torch
from torch import nn
import torch.nn.functional as F


FONT_DIR = Path(r"C:\Windows\Fonts")
CACHE_VERSION = "v2"
CACHE_PATH = (
    Path(__file__).with_name("__pycache__")
    / f"bookscribe_font_classifier_{CACHE_VERSION}.pt"
)
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 64
IMAGE_PADDING = 4
TRAINING_SIZES = (16, 20, 24, 30, 38, 48)
TRAINING_AUGMENTATIONS = 4
TRAINING_EPOCHS = 20
TRAINING_BATCH_SIZE = 128
TRAINING_LEARNING_RATE = 0.001
RANDOM_SEED = 1337
STYLES = ("regular", "bold", "italic", "bold_italic")
STYLE_LABELS = {style: index for index, style in enumerate(STYLES)}
CONFIDENCE_FLOOR = 0.55
ITALIC_CONFIDENCE_FLOOR = 0.72
ITALIC_MARGIN_FLOOR = 0.16
DESKEW_ANGLES = (-3.0, -2.25, -1.5, -0.75, 0.0, 0.75, 1.5, 2.25, 3.0)

TRAINING_TEXTS = (
    "The quick brown fox jumps over the lazy dog",
    "About the author and his memoirs",
    "Lee Kuan Yew is a statesman",
    "William Rees-Mogg, Editor of The Times",
    "The Singapore Story",
    "A deep and intense pro-Malay sentiment",
    "In the many years I have known him",
    "British Prime Minister, 1979-90",
    "Lee Kuan Yew",
    "George Bush",
    "Margaret Thatcher",
    "Henry Kissinger",
    "Singapore",
    "memoirs",
    "statesman",
    "visionary",
)


@dataclass(frozen=True)
class FontChoice:
    style: str
    font_name: str
    font_file: Path | None
    measure_font: fitz.Font


class FontClassifier:
    def __init__(self, family="times"):
        if family != "times":
            raise ValueError(f"Unsupported font family: {family}")

        self.family = family
        self.choices = _load_output_choices()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(RANDOM_SEED)
        self._model = _FontStyleNet().to(self._device)
        self._train_model()

    def classify(self, image):
        tensor = _image_tensor(image)
        if tensor is None:
            return self.choices["regular"]

        with torch.no_grad():
            logits = self._model(tensor.unsqueeze(0).to(self._device))
            probabilities = F.softmax(logits, dim=1).squeeze(0).cpu()

        confidence, label = torch.max(probabilities, dim=0)
        sorted_probabilities = torch.sort(probabilities, descending=True).values
        margin = float(sorted_probabilities[0] - sorted_probabilities[1])
        style = STYLES[int(label)]

        if _too_uncertain(style, float(confidence), margin):
            style = "regular"

        return self.choices[style]

    def measure_width(self, text, choice, font_size):
        return choice.measure_font.text_length(text, fontsize=font_size)

    def insert_kwargs(self, choice):
        kwargs = {"fontname": choice.font_name, "set_simple": 1}
        if choice.font_file is not None:
            kwargs["fontfile"] = str(choice.font_file)
        return kwargs

    def _train_model(self):
        if CACHE_PATH.is_file():
            state_dict = torch.load(CACHE_PATH, map_location=self._device, weights_only=True)
            self._model.load_state_dict(state_dict)
            self._model.eval()
            return

        torch.manual_seed(RANDOM_SEED)
        images, labels = _training_dataset()
        images = images.to(self._device)
        labels = labels.to(self._device)

        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=TRAINING_LEARNING_RATE,
            weight_decay=1e-4,
        )
        self._model.train()

        for _epoch in range(TRAINING_EPOCHS):
            permutation = torch.randperm(len(labels), device=self._device)
            for start in range(0, len(labels), TRAINING_BATCH_SIZE):
                batch_indices = permutation[start : start + TRAINING_BATCH_SIZE]
                logits = self._model(images[batch_indices])
                loss = F.cross_entropy(logits, labels[batch_indices])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self._model.eval()
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), CACHE_PATH)


class _FontStyleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(48, len(STYLES))

    def forward(self, inputs):
        features = self.features(inputs)
        return self.classifier(features.flatten(1))


def _training_dataset():
    rng = random.Random(RANDOM_SEED)
    images = []
    labels = []

    for font_set in _training_font_sets():
        for style, font_path in font_set.items():
            label = STYLE_LABELS[style]
            for size in TRAINING_SIZES:
                font = ImageFont.truetype(str(font_path), size=size)
                for text in TRAINING_TEXTS:
                    for _ in range(TRAINING_AUGMENTATIONS):
                        image = _render_training_text(text, font, style, rng)
                        tensor = _image_tensor(image)
                        if tensor is not None:
                            images.append(tensor)
                            labels.append(label)

    if not images:
        raise RuntimeError("No font training samples could be generated.")

    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def _training_font_sets():
    candidates = [
        {
            "regular": FONT_DIR / "times.ttf",
            "bold": FONT_DIR / "timesbd.ttf",
            "italic": FONT_DIR / "timesi.ttf",
            "bold_italic": FONT_DIR / "timesbi.ttf",
        },
        {
            "regular": FONT_DIR / "georgia.ttf",
            "bold": FONT_DIR / "georgiab.ttf",
            "italic": FONT_DIR / "georgiai.ttf",
            "bold_italic": FONT_DIR / "georgiaz.ttf",
        },
        {
            "regular": FONT_DIR / "cambria.ttc",
            "bold": FONT_DIR / "cambriab.ttf",
            "italic": FONT_DIR / "cambriai.ttf",
            "bold_italic": FONT_DIR / "cambriaz.ttf",
        },
        {
            "regular": FONT_DIR / "pala.ttf",
            "bold": FONT_DIR / "palab.ttf",
            "italic": FONT_DIR / "palai.ttf",
            "bold_italic": FONT_DIR / "palabi.ttf",
        },
    ]
    return [
        font_set
        for font_set in candidates
        if all(path.is_file() for path in font_set.values())
    ]


def _load_output_choices():
    windows_fonts = {
        "regular": FONT_DIR / "times.ttf",
        "bold": FONT_DIR / "timesbd.ttf",
        "italic": FONT_DIR / "timesi.ttf",
        "bold_italic": FONT_DIR / "timesbi.ttf",
    }

    if all(path.is_file() for path in windows_fonts.values()):
        return {
            style: FontChoice(
                style=style,
                font_name=f"bookscribe_times_{style}",
                font_file=path,
                measure_font=fitz.Font(fontfile=str(path)),
            )
            for style, path in windows_fonts.items()
        }

    built_ins = {
        "regular": "tiro",
        "bold": "tibo",
        "italic": "tiit",
        "bold_italic": "tibi",
    }
    return {
        style: FontChoice(
            style=style,
            font_name=font_name,
            font_file=None,
            measure_font=fitz.Font(font_name),
        )
        for style, font_name in built_ins.items()
    }


def _render_training_text(text, font, style, rng):
    left, top, right, bottom = font.getbbox(text)
    width = max(1, right - left + 24)
    height = max(1, bottom - top + 24)
    background = rng.randint(238, 255)
    ink = rng.randint(0, 35)
    image = Image.new("L", (width, height), background)
    draw = ImageDraw.Draw(image)
    draw.text((12 - left, 12 - top), text, fill=ink, font=font)

    if rng.random() < 0.7:
        image = image.rotate(rng.uniform(-1.2, 1.2), expand=True, fillcolor=background)
    if rng.random() < 0.55:
        image = _shear_image(image, _training_shear(style, rng), background)
    if rng.random() < 0.7:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.45)))
    if rng.random() < 0.7:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.75, 1.35))
    if rng.random() < 0.6:
        image = _add_noise(image, rng)

    return image


def _training_shear(style, rng):
    if style in ("regular", "bold"):
        return rng.uniform(-0.05, 0.05)

    return rng.uniform(-0.03, 0.03)


def _shear_image(image, shear, background):
    width, height = image.size
    x_shift = abs(shear) * height
    output_width = width + int(round(x_shift))
    offset = x_shift if shear > 0 else 0
    return image.transform(
        (output_width, height),
        Image.Transform.AFFINE,
        (1, -shear, offset, 0, 1, 0),
        Image.Resampling.BICUBIC,
        fillcolor=background,
    )


def _add_noise(image, rng):
    array = np.asarray(image, dtype=np.int16)
    noise = np.random.default_rng(rng.randint(0, 1_000_000)).normal(0, 3.0, array.shape)
    noisy = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="L")


def _image_tensor(image):
    prepared = _prepare_image(image)
    if prepared is None:
        return None

    array = np.asarray(prepared, dtype=np.float32) / 255.0
    ink = 1.0 - array
    return torch.from_numpy(ink).unsqueeze(0)


def _prepare_image(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    cropped = _crop_ink(gray, padding=3)
    if cropped is None:
        return None

    cropped = _deskew_text_image(cropped)
    cropped = ImageOps.autocontrast(cropped)
    cropped = _crop_ink(cropped, padding=3)
    if cropped is None:
        return None

    return _fit_to_canvas(cropped, IMAGE_WIDTH, IMAGE_HEIGHT)


def _fit_to_canvas(image, width, height):
    target_width = width - IMAGE_PADDING * 2
    target_height = height - IMAGE_PADDING * 2
    scale = min(target_width / image.width, target_height / image.height)
    resized_width = max(1, int(round(image.width * scale)))
    resized_height = max(1, int(round(image.height * scale)))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)

    canvas = Image.new("L", (width, height), 255)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def _crop_ink(gray, padding):
    ink = _ink_mask(gray)
    ys, xs = np.nonzero(ink)
    if len(xs) < 12:
        return None

    left = max(0, int(xs.min()) - padding)
    right = min(gray.width, int(xs.max()) + padding + 1)
    top = max(0, int(ys.min()) - padding)
    bottom = min(gray.height, int(ys.max()) + padding + 1)
    return gray.crop((left, top, right, bottom))


def _deskew_text_image(gray):
    if gray.width < 20 or gray.height < 8:
        return gray

    fill = int(np.percentile(np.asarray(gray, dtype=np.uint8), 92))
    best_image = gray
    best_score = _horizontal_ink_score(gray)

    for angle in DESKEW_ANGLES:
        if angle == 0.0:
            continue

        rotated = gray.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=fill,
        )
        score = _horizontal_ink_score(rotated)
        if score > best_score:
            best_score = score
            best_image = rotated

    return best_image


def _horizontal_ink_score(gray):
    ink = _ink_mask(gray)
    if int(ink.sum()) <= 0:
        return 0.0

    row_counts = ink.sum(axis=1).astype(np.float32)
    return float(row_counts.var() / max(1.0, row_counts.mean()))


def _ink_mask(gray):
    array = np.asarray(gray, dtype=np.uint8)
    threshold = min(235, max(35, int(array.mean() - array.std() * 0.15)))
    return array < threshold


def _too_uncertain(style, confidence, margin):
    if confidence < CONFIDENCE_FLOOR:
        return True

    if style in ("italic", "bold_italic"):
        return confidence < ITALIC_CONFIDENCE_FLOOR or margin < ITALIC_MARGIN_FLOOR

    return False
