"""Translate parsed ingredient lists to English via free online services.

Items are joined with newlines and sent together so the translator sees
context (in isolation "đường" becomes "road"; next to "bột mì" it becomes
"sugar"). Google is tried first, then MyMemory, and any chunk whose line count
comes back wrong is retried item by item.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from deep_translator import GoogleTranslator, MyMemoryTranslator

TARGET = "en"

# OCR language code -> translator language code.
GOOGLE_CODES: dict[str, str] = {"ch_sim": "zh-CN"}
MYMEMORY_CODES: dict[str, str] = {
    "vi": "vi-VN",
    "hi": "hi-IN",
    "mr": "mr-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "kn": "kn-IN",
    "bn": "bn-IN",
    "fr": "fr-FR",
    "de": "de-DE",
    "es": "es-ES",
    "pt": "pt-PT",
    "it": "it-IT",
    "ar": "ar-SA",
    "ch_sim": "zh-CN",
    "ja": "ja-JP",
    "ko": "ko-KR",
}

# Per-request character limits of the free endpoints, with headroom.
CHUNK_LIMITS = {"google": 4500, "mymemory": 450}

BACKENDS = ("auto", "google", "mymemory")
BACKEND_LABELS = {
    "auto": "Auto (Google, then MyMemory)",
    "google": "Google Translate",
    "mymemory": "MyMemory",
}


@dataclass(frozen=True)
class Translation:
    items: tuple[str, ...]
    backend: str


class TranslationError(RuntimeError):
    """Raised when every backend failed; `errors` maps backend -> message."""

    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__("; ".join(f"{name}: {msg}" for name, msg in errors.items()))


def pick_source_language(ocr_languages: Iterable[str]) -> str | None:
    """Return the OCR language to translate from, or None if only English is used.

    With several non-English languages the first is used; Google can also be
    told to auto-detect, which `_make_translator` handles.
    """
    non_english = [code for code in ocr_languages if code != TARGET]
    return non_english[0] if non_english else None


def _make_translator(backend: str, source: str, multiple_sources: bool):
    if backend == "google":
        src = "auto" if multiple_sources else GOOGLE_CODES.get(source, source)
        return GoogleTranslator(source=src, target=TARGET)
    if backend == "mymemory":
        return MyMemoryTranslator(source=MYMEMORY_CODES.get(source, source), target="en-GB")
    raise ValueError(f"Unknown backend {backend!r}")


def _chunk(items: Sequence[str], limit: int) -> list[list[str]]:
    chunks: list[list[str]] = [[]]
    size = 0
    for item in items:
        item_len = len(item) + 1
        if chunks[-1] and size + item_len > limit:
            chunks.append([])
            size = 0
        chunks[-1].append(item)
        size += item_len
    return chunks


def _clean(text: str) -> str:
    return " ".join(text.split()).strip()


def translate_chunked(items: Sequence[str], translate: Callable[[str], str], limit: int) -> list[str]:
    """Translate `items` with the given function, joining them per chunk.

    Oversized single items and chunks that come back with the wrong number of
    lines fall back to one call per item.
    """
    output: list[str] = []
    for chunk in _chunk(items, limit):
        if len(chunk) == 1 and len(chunk[0]) > limit:
            output.append(_clean(translate(chunk[0][:limit])))
            continue
        translated = translate("\n".join(chunk)) or ""
        lines = [_clean(line) for line in translated.split("\n") if line.strip()]
        if len(lines) == len(chunk):
            output.extend(lines)
        else:
            output.extend(_clean(translate(item) or item) for item in chunk)
    return output


def translate_items(
    items: Sequence[str],
    source: str,
    backend: str = "auto",
    multiple_sources: bool = False,
) -> Translation:
    """Translate `items` from `source` to English using the chosen backend."""
    if not items:
        return Translation(items=(), backend="none")
    order = ("google", "mymemory") if backend == "auto" else (backend,)
    errors: dict[str, str] = {}
    for name in order:
        try:
            translator = _make_translator(name, source, multiple_sources)
            result = translate_chunked(items, translator.translate, CHUNK_LIMITS[name])
            return Translation(items=tuple(result), backend=name)
        except Exception as exc:  # noqa: BLE001 - surface any backend failure and try the next
            errors[name] = f"{type(exc).__name__}: {str(exc).strip()[:160]}"
    raise TranslationError(errors)
