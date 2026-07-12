from dataclasses import dataclass
from pathlib import Path
import random
import string

import cv2
import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
import torch
from torch import nn
import torch.nn.functional as F


FONT_DIR = Path(r"C:\Windows\Fonts")
CACHE_VERSION = "v3"
CACHE_PATH = (
    Path(__file__).with_name("__pycache__")
    / f"bookscribe_font_classifier_{CACHE_VERSION}.pt"
)
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 64
IMAGE_PADDING = 4
TRAINING_SIZES = (16, 22, 30, 40)
TRAINING_AUGMENTATIONS = 1
TRAINING_EPOCHS = 20
TRAINING_BATCH_SIZE = 256
TRAINING_LEARNING_RATE = 0.001
RANDOM_SEED = 1337
STYLES = ("regular", "bold", "italic", "bold_italic")
STYLE_ATTRIBUTES = {
    "regular": (0.0, 0.0),
    "bold": (1.0, 0.0),
    "italic": (0.0, 1.0),
    "bold_italic": (1.0, 1.0),
}
DESKEW_ANGLES = (-3.0, -2.25, -1.5, -0.75, 0.0, 0.75, 1.5, 2.25, 3.0)
SLANT_SHEAR_MIN = 0.075
SLANT_SHEAR_MAX = 0.36
SLANT_GAIN_MIN = 1.15

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


@dataclass(frozen=True)
class FontEvidence:
    bold_probability: float
    italic_probability: float
    slant_shear: float
    slant_gain: float
    stroke_mean: float
    stroke_p90: float
    ink_ratio: float


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
        bold_probability, italic_probability = self.predict_attributes(image)
        style = _style_from_attributes(
            bold_probability >= 0.5,
            italic_probability >= 0.5,
        )
        return self.choices[style]

    def predict_attributes(self, image):
        tensor = _image_tensor(image)
        if tensor is None:
            return 0.0, 0.0

        with torch.no_grad():
            logits = self._model(tensor.unsqueeze(0).to(self._device))
            probabilities = torch.sigmoid(logits).squeeze(0).cpu()

        return float(probabilities[0]), float(probabilities[1])

    def classify_word_groups(self, word_groups, kinds):
        evidence_groups = [
            [self._word_evidence(image) for image in images]
            for images in word_groups
        ]
        styles = [
            ["bold" if kind == "title" else "regular"] * len(evidences)
            for evidences, kind in zip(evidence_groups, kinds)
        ]
        for group_index, (evidences, kind) in enumerate(zip(evidence_groups, kinds)):
            if kind == "title":
                _classify_title_group(styles[group_index], evidences)
                continue

            candidates = [_is_slanted(evidence) for evidence in evidences]
            candidates = _fill_single_false_gaps(candidates)
            candidates = _extend_runs_from_cnn(candidates, evidences)
            candidates = _fill_single_false_gaps(candidates)

            for start, end in _true_runs(candidates):
                if end - start < 2:
                    continue

                for word_index in range(start, end):
                    styles[group_index][word_index] = "italic"
                prefix_length = _bold_italic_prefix_length(evidences[start:end])
                for word_index in range(start, start + prefix_length):
                    styles[group_index][word_index] = "bold_italic"

        return [
            [self.choices[style] for style in group_styles]
            for group_styles in styles
        ]

    def _word_evidence(self, image):
        bold_probability, italic_probability = self.predict_attributes(image)
        slant_shear, slant_gain = _slant_evidence(image)
        stroke_mean, stroke_p90, ink_ratio = _stroke_evidence(image)
        return FontEvidence(
            bold_probability=bold_probability,
            italic_probability=italic_probability,
            slant_shear=slant_shear,
            slant_gain=slant_gain,
            stroke_mean=stroke_mean,
            stroke_p90=stroke_p90,
            ink_ratio=ink_ratio,
        )

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
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    labels[batch_indices],
                )
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
        self.classifier = nn.Linear(48, 2)

    def forward(self, inputs):
        features = self.features(inputs)
        return self.classifier(features.flatten(1))


def _training_dataset():
    rng = random.Random(RANDOM_SEED)
    images = []
    labels = []

    for font_set in _training_font_sets():
        for style, font_path in font_set.items():
            label = STYLE_ATTRIBUTES[style]
            for size in TRAINING_SIZES:
                font = ImageFont.truetype(str(font_path), size=size)
                for text in _training_words():
                    for _ in range(TRAINING_AUGMENTATIONS):
                        image = _render_training_text(text, font, style, rng)
                        tensor = _image_tensor(image)
                        if tensor is not None:
                            images.append(tensor)
                            labels.append(label)

    if not images:
        raise RuntimeError("No font training samples could be generated.")

    return torch.stack(images), torch.tensor(labels, dtype=torch.float32)


def _training_words():
    words = [
        word.strip(string.punctuation)
        for text in TRAINING_TEXTS
        for word in text.split()
    ]
    rng = random.Random(RANDOM_SEED)

    for _ in range(160):
        length = rng.randint(2, 14)
        word = "".join(rng.choice(string.ascii_letters) for _ in range(length))
        if rng.random() < 0.25:
            word += rng.choice((",", ".", ":", ";", "-93"))
        words.append(word)

    return tuple(dict.fromkeys(word for word in words if word))


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

    if rng.random() < 0.35:
        scale = rng.uniform(0.55, 0.9)
        reduced = image.resize(
            (
                max(1, int(round(image.width * scale))),
                max(1, int(round(image.height * scale))),
            ),
            Image.Resampling.BILINEAR,
        )
        image = reduced.resize(image.size, Image.Resampling.BILINEAR)

    morphology_roll = rng.random()
    if morphology_roll < 0.28:
        image = image.filter(ImageFilter.MinFilter(3))
    elif morphology_roll < 0.42:
        image = image.filter(ImageFilter.MaxFilter(3))

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


def _style_from_attributes(is_bold, is_italic):
    if is_bold and is_italic:
        return "bold_italic"
    if is_bold:
        return "bold"
    if is_italic:
        return "italic"
    return "regular"


def _is_slanted(evidence):
    return (
        evidence.ink_ratio >= 0.15
        and SLANT_SHEAR_MIN <= evidence.slant_shear <= SLANT_SHEAR_MAX
        and evidence.slant_gain >= SLANT_GAIN_MIN
    )


def _classify_title_group(styles, evidences):
    if not evidences:
        return

    is_italic = max(evidence.italic_probability for evidence in evidences) >= 0.9
    if not is_italic:
        return

    mean_bold = sum(evidence.bold_probability for evidence in evidences) / len(evidences)
    style = "bold_italic" if mean_bold >= 0.8 else "italic"
    for index in range(len(styles)):
        styles[index] = style


def _fill_single_false_gaps(values):
    result = list(values)
    for index in range(1, len(values) - 1):
        if not values[index] and values[index - 1] and values[index + 1]:
            result[index] = True
    return result


def _extend_runs_from_cnn(values, evidences):
    result = list(values)
    for start, end in _true_runs(values):
        if end - start < 2 or start == 0:
            continue
        previous = evidences[start - 1]
        if previous.italic_probability >= 0.7 or (
            previous.ink_ratio >= 0.15 and previous.slant_gain >= 1.05
        ):
            result[start - 1] = True
    return result


def _true_runs(values):
    runs = []
    start = None

    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None

    if start is not None:
        runs.append((start, len(values)))

    return runs


def _bold_italic_prefix_length(evidences):
    if len(evidences) < 4:
        return 0

    stroke_means = np.asarray(
        [evidence.stroke_mean for evidence in evidences],
        dtype=np.float32,
    )
    drops = stroke_means[:-1] - stroke_means[1:]
    if len(drops) > 1:
        drops = drops[:-1]
    largest_drop = float(drops.max()) if len(drops) else 0.0
    if largest_drop < 0.025:
        return 0

    near_largest = np.nonzero(drops >= largest_drop - 0.01)[0]
    return int(near_largest[-1]) + 1


def _stroke_evidence(image):
    gray = np.asarray(ImageOps.autocontrast(ImageOps.grayscale(image)), dtype=np.uint8)
    _threshold, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    distances = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ink_distances = distances[mask > 0]
    if not len(ink_distances):
        return 0.0, 0.0, 0.0

    return (
        float(ink_distances.mean()),
        float(np.quantile(ink_distances, 0.9)),
        float((mask > 0).mean()),
    )


def _slant_evidence(image):
    gray = np.asarray(ImageOps.autocontrast(ImageOps.grayscale(image)), dtype=np.uint8)
    _threshold, mask = cv2.threshold(
        gray,
        0,
        1,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    height, width = mask.shape
    if height < 4 or width < 2 or not int(mask.sum()):
        return 0.0, 1.0

    scores = []
    for shear in np.linspace(-0.4, 0.4, 33):
        extra_width = int(round(abs(shear) * height)) + 4
        transform = np.float32(
            [
                [1.0, shear, extra_width / 2.0],
                [0.0, 1.0, 2.0],
            ]
        )
        transformed = cv2.warpAffine(
            mask,
            transform,
            (width + extra_width, height + 4),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        column_counts = transformed.sum(axis=0).astype(np.float32)
        score = float(column_counts.var() / max(1.0, column_counts.mean()))
        scores.append((float(shear), score))

    best_shear, best_score = max(scores, key=lambda item: item[1])
    zero_score = min(scores, key=lambda item: abs(item[0]))[1]
    return best_shear, best_score / max(zero_score, 1e-6)
