import pytest

import translate
from translate import TranslationError, pick_source_language, translate_chunked, translate_items


def test_pick_source_language():
    assert pick_source_language(("en", "vi")) == "vi"
    assert pick_source_language(("en",)) is None
    assert pick_source_language(("en", "vi", "fr")) == "vi"


def test_translate_chunked_joins_items_and_splits_lines():
    calls = []

    def fake(text):
        calls.append(text)
        return "\n".join(f"EN({line})" for line in text.split("\n"))

    out = translate_chunked(["đường", "muối", "bột mì"], fake, limit=100)
    assert out == ["EN(đường)", "EN(muối)", "EN(bột mì)"]
    assert calls == ["đường\nmuối\nbột mì"]


def test_translate_chunked_respects_limit_and_falls_back_on_line_mismatch():
    calls = []

    def fake(text):
        calls.append(text)
        # Collapse newlines, simulating a service that merges lines.
        return "MERGED " + text.replace("\n", " ")

    out = translate_chunked(["aaaa", "bbbb", "cccc"], fake, limit=10)
    # Chunks: ["aaaa", "bbbb"] (10 chars incl. separators) then ["cccc"].
    assert out == ["MERGED aaaa", "MERGED bbbb", "MERGED cccc"]
    assert calls[0] == "aaaa\nbbbb"
    assert calls[1:3] == ["aaaa", "bbbb"]


def test_translate_items_falls_back_to_next_backend(monkeypatch):
    class Broken:
        def translate(self, text):
            raise RuntimeError("blocked")

    class Works:
        def translate(self, text):
            return text.upper()

    def fake_make(backend, source, multiple_sources):
        return Broken() if backend == "google" else Works()

    monkeypatch.setattr(translate, "_make_translator", fake_make)
    result = translate_items(["đường", "muối"], "vi", backend="auto")
    assert result.backend == "mymemory"
    assert result.items == ("ĐƯỜNG", "MUỐI")


def test_translate_items_raises_with_all_errors(monkeypatch):
    class Broken:
        def translate(self, text):
            raise RuntimeError("blocked")

    monkeypatch.setattr(translate, "_make_translator", lambda *a, **k: Broken())
    with pytest.raises(TranslationError) as info:
        translate_items(["x"], "vi", backend="auto")
    assert set(info.value.errors) == {"google", "mymemory"}


def test_translate_items_empty_is_noop():
    assert translate_items([], "vi").items == ()
