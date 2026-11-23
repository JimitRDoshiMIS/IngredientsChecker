from PIL import Image

from ocr import (
    OcrResult,
    draw_boxes,
    extract_ingredients,
    fold_text,
    normalise_languages,
    parse_raw_results,
    prepare_image,
    results_to_text,
    validate_languages,
)


def box(x, y, w, h):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def test_normalise_languages_always_includes_english_and_drops_unknown():
    assert normalise_languages(["hi", "xx", "hi"]) == ["en", "hi"]
    assert normalise_languages([]) == ["en"]


def test_validate_languages_latin_mix_is_fine():
    assert validate_languages(["en", "vi", "fr"]) is None
    assert validate_languages(["en", "hi"]) is None


def test_validate_languages_rejects_unsupported_combinations():
    assert "Hindi" in validate_languages(["en", "vi", "hi"])
    assert "only one non-latin" in validate_languages(["en", "hi", "ta"]).lower()


def test_fold_text_strips_accents_and_keeps_length():
    original = "Thành phần: Đường, Sữa"
    folded = fold_text(original)
    assert folded == "thanh phan: đuong, sua"
    assert len(folded) == len(original)


def test_extract_ingredients_english_header_terminator_and_nested_brackets():
    text = (
        "Net Wt 420 g\n"
        "INGREDIENTS: Potato (92%), Palm Oil, Chocolate (Sugar, Cocoa Butter), Salt.\n"
        "Allergen advice: may contain milk."
    )
    assert extract_ingredients(text) == [
        "Potato (92%)",
        "Palm Oil",
        "Chocolate (Sugar, Cocoa Butter)",
        "Salt",
    ]


def test_extract_ingredients_vietnamese_header():
    text = "Thành phần: Đường, sữa bột, bột cacao (12%); hương liệu tổng hợp. Bảo quản nơi khô ráo."
    assert extract_ingredients(text) == ["Đường", "sữa bột", "bột cacao (12%)", "hương liệu tổng hợp"]


def test_extract_ingredients_bilingual_header_chain():
    text = "THÀNH PHẦN / INGREDIENTS: Bột mì (62%), đường, muối. Dị ứng: có thể chứa trứng."
    assert extract_ingredients(text) == ["Bột mì (62%)", "đường", "muối"]


def test_extract_ingredients_without_header_needs_comma_list():
    assert extract_ingredients("Hello world") == []
    assert extract_ingredients("Sugar, salt, water, flour") == ["Sugar", "salt", "water", "flour"]


def test_parse_raw_results_filters_blank_and_low_confidence():
    raw = [
        (box(0, 0, 10, 10), "  hello ", 0.9),
        (box(0, 20, 10, 10), "   ", 0.9),
        (box(0, 40, 10, 10), "faint", 0.2),
    ]
    results = parse_raw_results(raw, min_confidence=0.5)
    assert [r.text for r in results] == ["hello"]
    assert results[0].bbox == ((0, 0), (10, 0), (10, 10), (0, 10))


def test_results_to_text_orders_lines_and_words():
    results = parse_raw_results(
        [
            (box(120, 52, 40, 20), "world", 0.9),
            (box(10, 100, 40, 20), "second", 0.9),
            (box(10, 50, 40, 20), "Hello", 0.9),
            (box(60, 102, 40, 20), "line", 0.9),
        ]
    )
    assert results_to_text(results) == "Hello world\nsecond line"


def test_results_to_text_empty():
    assert results_to_text([]) == ""


def test_prepare_image_downscales_and_converts_to_rgb():
    image = Image.new("RGBA", (4000, 2000))
    prepared = prepare_image(image)
    assert prepared.mode == "RGB"
    assert max(prepared.size) == 2000
    assert prepared.size == (2000, 1000)


def test_prepare_image_enhance_produces_grayscale_rgb():
    image = Image.new("RGB", (50, 50), (200, 40, 40))
    enhanced = prepare_image(image, enhance=True)
    assert enhanced.mode == "RGB"
    r, g, b = enhanced.getpixel((25, 25))
    assert r == g == b


def test_prepare_image_rotates_clockwise():
    image = Image.new("RGB", (40, 20), "white")
    image.putpixel((0, 0), (255, 0, 0))  # top-left marker
    rotated = prepare_image(image, rotation=90)
    assert rotated.size == (20, 40)
    # A clockwise quarter turn moves the top-left corner to the top-right.
    assert rotated.getpixel((19, 0)) == (255, 0, 0)


def test_run_ocr_caps_detector_canvas_and_applies_small_text():
    import ocr

    class FakeReader:
        def __init__(self):
            self.kwargs = None

        def readtext(self, array, **kwargs):
            self.kwargs = kwargs
            return [(box(0, 0, 10, 10), "x", 0.9)]

    reader = FakeReader()
    results = ocr.run_ocr(reader, Image.new("RGB", (20, 20)), small_text=True)
    assert reader.kwargs["canvas_size"] == ocr.DETECTOR_CANVAS
    assert reader.kwargs["mag_ratio"] == 1.5
    assert [r.text for r in results] == ["x"]


def test_draw_boxes_does_not_mutate_original():
    image = Image.new("RGB", (100, 100), "white")
    result = OcrResult(bbox=((10, 10), (50, 10), (50, 30), (10, 30)), text="x", confidence=1.0)
    annotated = draw_boxes(image, [result])
    assert annotated is not image
    assert image.getpixel((10, 10)) == (255, 255, 255)
    assert annotated.getpixel((10, 10)) != (255, 255, 255)
