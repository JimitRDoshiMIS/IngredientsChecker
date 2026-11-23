"""OCR helpers built on EasyOCR.

The heavy EasyOCR reader is created lazily and cached so the model weights are
loaded only once per process.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

# Language code -> human label. EasyOCR requires every non-Latin language to be
# paired with English, so "en" is always included in the reader.
SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "vi": "Vietnamese",
    "hi": "Hindi",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "bn": "Bengali",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "ar": "Arabic",
    "ch_sim": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
}

MAX_IMAGE_SIDE = 2000

# Longest side (px) the text *detector* is fed. This dominates peak memory:
# 2560 -> ~4 GB, 1600 -> ~2.3 GB, 1280 -> ~1.5 GB, 1024 -> ~1.1 GB. Recognition
# still crops from the full-resolution image, so quality holds. Override with
# the OCR_CANVAS_SIZE environment variable on machines with more RAM.
DETECTOR_CANVAS = int(os.environ.get("OCR_CANVAS_SIZE", "1280"))

DEFAULT_LANGUAGES: tuple[str, ...] = ("en", "vi")

# Latin-script languages share one EasyOCR recognition model and can be mixed
# freely. Any other script can only be paired with English.
LATIN_SCRIPT_LANGUAGES = frozenset({"en", "vi", "fr", "de", "es", "pt", "it"})

# Header phrases that introduce an ingredient list (accent-folded, lower-case).
INGREDIENT_HEADERS = (
    "ingredients",
    "ingredient",
    "composition",
    "thanh phan",  # thành phần
    "nguyen lieu",  # nguyên liệu
)

# Phrases that usually follow the ingredient list on a label.
INGREDIENT_TERMINATORS = (
    "allergen",
    "allergy",
    "contains",
    "may contain",
    "nutrition",
    "storage",
    "store in",
    "best before",
    "manufactured",
    "net wt",
    "net weight",
    "di ung",  # dị ứng
    "co the chua",  # có thể chứa
    "gia tri dinh duong",  # giá trị dinh dưỡng
    "bao quan",  # bảo quản
    "huong dan",  # hướng dẫn
    "san xuat",  # sản xuất
    "khoi luong",  # khối lượng
    "hsd",
    "nsx",
)

BBox = Sequence[Sequence[float]]


@dataclass(frozen=True)
class OcrResult:
    """A single detected text region."""

    bbox: tuple[tuple[int, int], ...]
    text: str
    confidence: float


def normalise_languages(languages: Iterable[str]) -> list[str]:
    """Return a sorted, de-duplicated language list that always contains English."""
    langs = {code for code in languages if code in SUPPORTED_LANGUAGES}
    langs.add("en")
    return sorted(langs)


def validate_languages(languages: Iterable[str]) -> str | None:
    """Return an error message if EasyOCR cannot load this language combination."""
    langs = set(languages)
    non_latin = sorted(langs - LATIN_SCRIPT_LANGUAGES)
    other_latin = sorted(langs & LATIN_SCRIPT_LANGUAGES - {"en"})
    if len(non_latin) > 1:
        names = ", ".join(SUPPORTED_LANGUAGES[c] for c in non_latin)
        return f"Only one non-Latin script can be used at a time. Pick just one of: {names}."
    if non_latin and other_latin:
        return (
            f"{SUPPORTED_LANGUAGES[non_latin[0]]} can only be combined with English. "
            f"Deselect {', '.join(SUPPORTED_LANGUAGES[c] for c in other_latin)} or {SUPPORTED_LANGUAGES[non_latin[0]]}."
        )
    return None


def enhance_label(image: Image.Image) -> Image.Image:
    """Boost legibility of small, low-contrast print on packaging.

    Grayscale removes coloured backgrounds, autocontrast stretches faded ink and a
    gentle unsharp mask crispens tiny glyph edges.
    """
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))
    return gray.convert("RGB")


def prepare_image(image: Image.Image, rotation: int = 0, enhance: bool = False) -> Image.Image:
    """Fix EXIF orientation, apply an optional clockwise rotation, convert to RGB,
    cap the size for faster OCR and optionally enhance for small label text."""
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    rotation %= 360
    if rotation:
        # PIL rotates counter-clockwise; negate for a clockwise user-facing control.
        image = image.rotate(-rotation, expand=True)
    longest = max(image.size)
    if longest > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / longest
        new_size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    if enhance:
        image = enhance_label(image)
    return image


def parse_raw_results(raw: Iterable[tuple[BBox, str, float]], min_confidence: float = 0.0) -> list[OcrResult]:
    """Convert EasyOCR's raw `(bbox, text, confidence)` tuples into OcrResult objects."""
    results: list[OcrResult] = []
    for bbox, text, confidence in raw:
        text = text.strip()
        if not text or confidence < min_confidence:
            continue
        points = tuple((int(round(x)), int(round(y))) for x, y in bbox)
        results.append(OcrResult(bbox=points, text=text, confidence=float(confidence)))
    return results


def results_to_text(results: Sequence[OcrResult], line_tolerance: float = 0.6) -> str:
    """Assemble detected regions into reading order (top-to-bottom, left-to-right).

    Regions whose vertical centres are within `line_tolerance` * median box
    height of each other are joined on the same line with a space.
    """
    if not results:
        return ""

    def centre_y(r: OcrResult) -> float:
        return sum(p[1] for p in r.bbox) / len(r.bbox)

    def height(r: OcrResult) -> float:
        ys = [p[1] for p in r.bbox]
        return max(ys) - min(ys)

    def left_x(r: OcrResult) -> float:
        return min(p[0] for p in r.bbox)

    heights = sorted(height(r) for r in results)
    median_height = heights[len(heights) // 2] or 1.0
    threshold = median_height * line_tolerance

    ordered = sorted(results, key=centre_y)
    lines: list[list[OcrResult]] = [[ordered[0]]]
    for region in ordered[1:]:
        current_line = lines[-1]
        line_centre = sum(centre_y(r) for r in current_line) / len(current_line)
        if abs(centre_y(region) - line_centre) <= threshold:
            current_line.append(region)
        else:
            lines.append([region])

    return "\n".join(" ".join(r.text for r in sorted(line, key=left_x)) for line in lines)


def draw_boxes(image: Image.Image, results: Sequence[OcrResult]) -> Image.Image:
    """Return a copy of `image` with each detected region outlined."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width = max(2, round(min(annotated.size) / 300))
    for result in results:
        draw.polygon(result.bbox, outline=(255, 75, 75), width=width)
    return annotated


def _fold_char(char: str) -> str:
    """Map a character to its lower-case, accent-free base, keeping length 1."""
    base = unicodedata.normalize("NFD", char)[0]
    if unicodedata.combining(base):
        return char
    lowered = base.lower()
    return lowered if len(lowered) == 1 else base


def fold_text(text: str) -> str:
    """Accent-fold and lower-case `text` without changing its length, so indices
    found in the folded string map back onto the original."""
    return "".join(_fold_char(c) for c in text)


def _split_top_level(text: str) -> list[str]:
    """Split on commas/semicolons that are not inside (), [] or {}."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}" and depth:
            depth -= 1
        if char in ",;" and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return parts


def extract_ingredients(text: str) -> list[str]:
    """Pull an ingredient list out of OCR'd label text.

    Looks for an English or Vietnamese "Ingredients:" header, stops at the next
    label section (allergens, storage, nutrition, ...) and splits on top-level
    commas/semicolons so sub-ingredients in brackets stay attached. Without a
    header, falls back to the whole text when it looks like a comma list.
    """
    flat = " ".join(text.split())
    if not flat:
        return []
    folded = fold_text(flat)

    # Labels are often bilingual ("THÀNH PHẦN / INGREDIENTS:"), so consume a
    # whole chain of headers and separators rather than just the first one.
    header_alt = "|".join(re.escape(h) for h in INGREDIENT_HEADERS)
    header_re = re.compile(rf"(?:\b(?:{header_alt})\b\s*[:：.\-]?\s*(?:[/|]\s*)?)+")
    match = header_re.search(folded)
    header_found = match is not None
    start = match.end() if match else 0

    body = folded[start:]
    end = len(body)
    for terminator in INGREDIENT_TERMINATORS:
        match = re.search(rf"\b{re.escape(terminator)}\b", body)
        if match and 0 < match.start() < end:
            end = match.start()

    segment = flat[start : start + end]
    items = [item.strip(" .:;-–") for item in _split_top_level(segment)]
    items = [item for item in items if len(item) > 1]
    if not header_found and len(items) < 3:
        return []
    return items


def split_all_text(text: str) -> list[str]:
    """Fallback when no ingredient list is found: split every OCR line on
    top-level commas/semicolons so the diet checks can still run over all text."""
    items: list[str] = []
    for line in text.splitlines():
        for item in _split_top_level(line):
            item = item.strip(" .:;-–")
            if len(item) > 1:
                items.append(item)
    return items


def run_ocr(
    reader,
    image: Image.Image,
    min_confidence: float = 0.0,
    paragraph: bool = False,
    auto_rotate: bool = False,
    small_text: bool = False,
) -> list[OcrResult]:
    """Run EasyOCR on a PIL image and return parsed results.

    With `auto_rotate`, EasyOCR also tries each text box at 90/180/270 degrees
    and keeps the most confident reading (roughly 4x slower). `small_text`
    upsamples the image before detection so fine label print is found.
    """
    array = np.asarray(image)
    kwargs = {"paragraph": paragraph, "canvas_size": DETECTOR_CANVAS}
    if auto_rotate:
        kwargs["rotation_info"] = [90, 180, 270]
    if small_text:
        kwargs["mag_ratio"] = 1.5
    raw = reader.readtext(array, **kwargs)
    if paragraph:
        # In paragraph mode EasyOCR omits the confidence value.
        raw = [(bbox, text, 1.0) for bbox, text in raw]
    return parse_raw_results(raw, min_confidence=min_confidence)
