from diet import classify_ingredient, classify_ingredients, overall_status


def statuses(*items):
    return [classify_ingredient(i).status for i in items]


def test_egg_and_egg_derivatives_are_non_veg():
    assert statuses("Egg", "whole eggs", "egg yolk powder", "albumen", "lysozyme") == ["non_veg"] * 5
    assert classify_ingredient("Eggs").matches == ("egg",)


def test_eggplant_is_vegetarian():
    assert classify_ingredient("eggplant").status == "veg"


def test_dairy_and_honey_are_vegetarian():
    assert statuses("Milk powder", "butter", "whey", "goat milk", "honey", "cheese") == ["veg"] * 6


def test_english_meat_and_seafood():
    assert statuses("Chicken extract", "fish sauce", "Gelatin", "Beef", "shrimp paste", "carmine") == ["non_veg"] * 6


def test_e_numbers():
    assert classify_ingredient("colour (120)").status == "non_veg"
    assert classify_ingredient("colour (E120)").status == "non_veg"
    assert classify_ingredient("emulsifier (471)").status == "uncertain"
    assert classify_ingredient("emulsifier (322 from soybeans)").status == "veg"
    assert classify_ingredient("120g").status == "veg"


def test_vietnamese_is_accent_sensitive():
    # bơ = butter (veg) vs bò = beef (non-veg); ca cao vs cá (fish)
    assert classify_ingredient("bơ").status == "veg"
    assert classify_ingredient("bò").status == "non_veg"
    assert classify_ingredient("bột ca cao").status == "veg"
    assert classify_ingredient("cá cơm").status == "non_veg"
    assert classify_ingredient("xương rồng").status == "veg"
    assert classify_ingredient("nước hầm xương").status == "non_veg"


def test_vietnamese_common_non_veg_terms():
    assert statuses("trứng gà", "nước mắm", "dầu hào", "mắm tôm", "thịt heo", "mỡ heo") == ["non_veg"] * 6
    assert "egg (trứng)" in classify_ingredient("trứng").matches


def test_translation_is_also_checked():
    # OCR dropped the diacritics; the English translation still catches it.
    check = classify_ingredient("trung ga", "chicken egg")
    assert check.status == "non_veg"
    assert set(check.matches) >= {"egg", "poultry"}


def test_matches_are_deduplicated_by_category():
    check = classify_ingredient("trứng gà", "hen eggs")
    assert check.matches.count("egg") == 1
    assert not any(m.startswith("egg (") for m in check.matches)


def test_uncertain_additives():
    assert classify_ingredient("mono- and diglycerides of fatty acids").status == "uncertain"
    assert classify_ingredient("rennet").status == "uncertain"
    # Meat flavourings are treated conservatively as non-veg.
    assert classify_ingredient("hương gà").status == "non_veg"


def test_overall_status_precedence():
    checks = classify_ingredients(["sugar", "glycerin", "salt"])
    assert overall_status(checks) == "uncertain"
    checks = classify_ingredients(["sugar", "glycerin", "egg"])
    assert overall_status(checks) == "non_veg"
    assert overall_status(classify_ingredients(["sugar", "salt"])) == "veg"
    assert overall_status([]) == "veg"


def test_classify_ingredients_ignores_mismatched_translation_length():
    checks = classify_ingredients(["trứng"], ["a", "b"])
    assert checks[0].status == "non_veg"
