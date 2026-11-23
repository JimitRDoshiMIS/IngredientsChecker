"""Vegetarian / non-vegetarian classification of ingredient names.

Each ingredient is checked against English patterns (on accent-folded,
lower-cased text) and Vietnamese patterns (lower-cased but accent-sensitive,
because e.g. "bò" = beef while "bơ" = butter). Egg and all egg derivatives are
treated as non-vegetarian. Additives that are *sometimes* animal-derived are
reported as "uncertain" rather than non-vegetarian.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from ocr import fold_text

Status = Literal["non_veg", "uncertain", "veg"]

# (regex on folded English text, label shown to the user)
NON_VEG_EN: tuple[tuple[str, str], ...] = (
    (r"\beggs?\b", "egg"),
    (r"\begg (yolks?|whites?|powder|solids?|lecithin)\b", "egg"),
    (r"\b(albumen|albumin|ovalbumin|lysozyme|egg albumen)\b", "egg derivative"),
    (r"\bmeats?\b", "meat"),
    (r"\b(beef|veal|oxtail)\b", "beef"),
    (r"\b(pork|bacon|ham|lard|pancetta|chorizo|pepperoni|salami)\b", "pork"),
    (r"\b(chicken|poultry|turkey|duck|goose|quail)\b", "poultry"),
    (r"\b(lamb|mutton|venison|rabbit)\b", "meat"),
    (r"\bgoat(?!'?s? ?(milk|cheese|yogh?urt|butter))\b", "meat"),
    (r"\b(sausages?|hot ?dogs?|meat ?balls?|pate|liver|offal|tripe|bone ?broth|bone ?meal)\b", "meat"),
    (r"\b(tallow|suet|dripping|schmaltz)\b", "animal fat"),
    (r"\b(fish|anchov(y|ies)|sardines?|tuna|salmon|cod|mackerel|herring|tilapia|catfish|pollock)\b", "fish"),
    (r"\b(fish|anchovy|oyster|shrimp|prawn|crab) ?(sauce|paste|extract|powder|oil|stock)\b", "seafood"),
    (r"\b(shrimps?|prawns?|crabs?|lobsters?|squid|cuttlefish|octopus|clams?|mussels?|scallops?|oysters?(?! mushrooms?))\b", "seafood"),
    (r"\b(seafood|shellfish|krill|roe|caviar|bonito|katsuobushi|dashi|surimi)\b", "seafood"),
    (r"\b(gelatine?|isinglass|collagen)\b", "gelatin"),
    (r"\b(carmine|cochineal|carminic acid)\b", "carmine (insect dye)"),
    (r"(?<![\w.])(e ?|ins ?)?(120|441|542|631|627|635)(?![\w])", "animal-derived additive"),
)

UNCERTAIN_EN: tuple[tuple[str, str], ...] = (
    (r"\b(rennet|enzymes?)\b", "rennet/enzymes (may be animal)"),
    (r"\b(l-?cysteine)\b", "L-cysteine (may be animal)"),
    (r"\b(shellac|confectioner'?s glaze)\b", "shellac (insect secretion)"),
    (r"\b(mono-? ?and di-?glycerides|monoglycerides|diglycerides)\b", "E471 (may be animal fat)"),
    (r"\b(glycerine?|glycerol)\b", "glycerin (may be animal fat)"),
    (r"\b(stearic acid|magnesium stearate|stearates?|stearoyl)\b", "stearate (may be animal fat)"),
    (r"\b(omega-?3|vitamin d3|cholecalciferol)\b", "may be fish/lanolin derived"),
    (r"(?<![\w.])(e ?|ins ?)?(422|471|472[a-f]?|481|482|570|904|920)(?![\w])", "additive (may be animal)"),
)

# (regex on lower-cased, accent-preserved Vietnamese text, label)
NON_VEG_VI: tuple[tuple[str, str], ...] = (
    (r"\btrứng\b", "egg (trứng)"),
    (r"\blòng (đỏ|trắng)\b", "egg (lòng đỏ/trắng)"),
    (r"\bthịt\b", "meat (thịt)"),
    (r"\bbò\b", "beef (bò)"),
    (r"\b(heo|lợn|ba rọi|ba chỉ|giò|chả lụa|xúc xích|lạp xưởng)\b", "pork (heo/lợn)"),
    (r"\bmỡ\b", "animal fat (mỡ)"),
    (r"\b(gà|vịt|ngan|ngỗng|chim cút)\b", "poultry (gà/vịt)"),
    (r"\b(cừu|dê|nai|thỏ)\b", "meat"),
    (r"\b(gan|lòng|cật|da heo)\b", "meat/offal"),
    (r"\bxương(?! rồng)\b", "bone (xương)"),
    (r"\bcá\b", "fish (cá)"),
    (r"\b(cá cơm|cá ngừ|cá hồi|cá thu|cá basa|cá tra|cá khô|khô cá|ruốc cá|chà bông)\b", "fish"),
    (r"\bnước mắm\b", "fish sauce (nước mắm)"),
    (r"\b(mắm|mắm tôm|mắm ruốc|mắm nêm|mắm cá)\b", "fermented seafood (mắm)"),
    (r"\b(tôm|tép|ruốc|cua|ghẹ|mực|bạch tuộc|sò|nghêu|ngao|hến|ốc|hàu|hào|bào ngư|lươn)\b", "seafood"),
    (r"\bdầu hào\b", "oyster sauce (dầu hào)"),
    (r"\bhải sản\b", "seafood (hải sản)"),
    (r"\b(gelatin|gelatine|thạch động vật|da cá|bong bóng cá)\b", "gelatin"),
    (r"\b(động vật)\b", "animal-derived (động vật)"),
)

UNCERTAIN_VI: tuple[tuple[str, str], ...] = (
    (r"\b(men dịch vị|rennet|enzym)\b", "rennet/enzymes (may be animal)"),
    (r"\bglycerin\b", "glycerin (may be animal fat)"),
)


def _compile(rules: Iterable[tuple[str, str]]) -> list[tuple[re.Pattern[str], str]]:
    return [(re.compile(pattern), label) for pattern, label in rules]


_NON_VEG_EN = _compile(NON_VEG_EN)
_UNCERTAIN_EN = _compile(UNCERTAIN_EN)
_NON_VEG_VI = _compile(NON_VEG_VI)
_UNCERTAIN_VI = _compile(UNCERTAIN_VI)


@dataclass(frozen=True)
class DietCheck:
    status: Status
    matches: tuple[str, ...]

    @property
    def emoji(self) -> str:
        return {"non_veg": "🔴", "uncertain": "🟠", "veg": "🟢"}[self.status]

    @property
    def label(self) -> str:
        return {"non_veg": "Non-vegetarian", "uncertain": "Uncertain", "veg": "Vegetarian"}[self.status]


def _vi_text(text: str) -> str:
    return unicodedata.normalize("NFC", text).lower()


def _find(text_en: str, text_vi: str, en_rules, vi_rules) -> list[str]:
    """Return matched labels, de-duplicated by category (the part before ' (')."""
    labels: list[str] = []
    seen: set[str] = set()
    for text, rules in ((text_en, en_rules), (text_vi, vi_rules)):
        for pattern, label in rules:
            key = label.split(" (")[0]
            if key not in seen and pattern.search(text):
                seen.add(key)
                labels.append(label)
    return labels


def classify_ingredient(original: str, english: str | None = None) -> DietCheck:
    """Classify one ingredient using its original text and optional English translation."""
    texts = [original] + ([english] if english else [])
    folded = " | ".join(fold_text(t) for t in texts)
    vi = " | ".join(_vi_text(t) for t in texts)

    non_veg = _find(folded, vi, _NON_VEG_EN, _NON_VEG_VI)
    if non_veg:
        return DietCheck("non_veg", tuple(non_veg))
    uncertain = _find(folded, vi, _UNCERTAIN_EN, _UNCERTAIN_VI)
    if uncertain:
        return DietCheck("uncertain", tuple(uncertain))
    return DietCheck("veg", ())


def classify_ingredients(originals: Sequence[str], english: Sequence[str] | None = None) -> list[DietCheck]:
    english = english if english and len(english) == len(originals) else [None] * len(originals)
    return [classify_ingredient(o, e) for o, e in zip(originals, english)]


def overall_status(checks: Iterable[DietCheck]) -> Status:
    statuses = {c.status for c in checks}
    if "non_veg" in statuses:
        return "non_veg"
    if "uncertain" in statuses:
        return "uncertain"
    return "veg"


# ---------------------------------------------------------------------------
# Jain diet: lacto-vegetarian plus no honey, alcohol, fungi, figs or anything
# that grows underground (onion, garlic, potato, carrot, radish, ...).
# Sources: Young Jains of America dietary guidance; jainknowledge.com "Jain
# foods list"; thejainreligion.in "Jain food list". Items whose avoidance
# varies between households are reported as "uncertain".
# ---------------------------------------------------------------------------

JainStatus = Literal["not_jain", "uncertain", "jain"]

NOT_JAIN_EN: tuple[tuple[str, str], ...] = (
    (r"\b(onions?|shallots?|scallions?|spring onions?|green onions?|leeks?|chives?|garlic|alliums?)\b", "onion/garlic"),
    (r"\b(potato(es)?|sweet potato(es)?)\b", "potato"),
    (r"\bcarrots?\b", "carrot"),
    (r"\bbeet(root)?s?\b(?! sugar)", "beetroot"),
    (r"\b(radish(es)?|daikon|turnips?|parsnips?|rutabagas?|swedes?|celeriac|salsify|horseradish|wasabi)\b", "root vegetable"),
    (r"\b(yams?|taro|arrowroot|konjac|jicama|lotus roots?|burdock|ginseng|liquorice|licorice|galangal)\b", "root/tuber"),
    (r"\b(mushrooms?|shiitake|porcini|enoki|portobello|champignons?|fung(us|i)|truffle oil)\b", "mushroom/fungus"),
    (r"\b(honey|mead)\b", "honey"),
    (r"\b(alcohol|ethanol|ethyl alcohol|wine(?! vinegar)|beer|rum|brandy|whisk(e)?y|vodka|liqueur|sake|mirin)\b", "alcohol"),
    (r"\b(figs?|anjeer)\b", "fig"),
)

JAIN_UNCERTAIN_EN: tuple[tuple[str, str], ...] = (
    (r"\b(yeast|leaven(ing)?)\b", "yeast (many Jains avoid)"),
    (r"\bvinegar\b", "vinegar (fermented; varies)"),
    (r"\b(fermented|soya? sauce|miso|tempeh|natto|kimchi|kombucha|kefir|sourdough)\b", "fermented (varies)"),
    (r"\bgingers?\b", "ginger (fresh avoided; dried varies)"),
    (r"\b(turmeric|curcumin)\b", "turmeric (varies)"),
    (r"\b(tapioca|sago|cassava|manioc|sabudana)\b", "tapioca/cassava root starch (varies)"),
    (r"\b(eggplants?|brinjals?|aubergines?)\b", "eggplant (multi-seeded; some avoid)"),
    (r"\b(asafoetida|asafetida|hing)\b", "asafoetida (some avoid)"),
    (r"\b(inulin|chicory)\b", "chicory root fibre (varies)"),
    (r"\bsprout(s|ed)?\b", "sprouts (varies)"),
)

NOT_JAIN_VI: tuple[tuple[str, str], ...] = (
    (r"\b(hành|hẹ|tỏi|boa rô|kiệu)\b", "onion/garlic (hành/tỏi)"),
    (r"\bkhoai(?! mì)\b", "potato/tuber (khoai)"),
    (r"\bcà rốt\b", "carrot (cà rốt)"),
    (r"\bcủ\b", "root/tuber (củ)"),
    (r"\b(riềng|sâm|cam thảo)\b", "root (riềng/sâm/cam thảo)"),
    (r"\bnấm\b", "mushroom (nấm)"),
    (r"\bmật ong\b", "honey (mật ong)"),
    (r"\b(rượu|bia|cồn)\b", "alcohol (rượu/bia/cồn)"),
    (r"\b(quả sung|trái sung)\b", "fig (sung)"),
)

JAIN_UNCERTAIN_VI: tuple[tuple[str, str], ...] = (
    (r"\bmen\b", "yeast (men; many avoid)"),
    (r"\bgiấm\b", "vinegar (giấm; varies)"),
    (r"\b(lên men|nước tương|xì dầu|tương|chao|kim chi)\b", "fermented (varies)"),
    (r"\bgừng\b", "ginger (gừng; varies)"),
    (r"\bnghệ\b", "turmeric (nghệ; varies)"),
    (r"\b(bột sắn|tinh bột sắn|bột năng|sắn|khoai mì|bột báng)\b", "tapioca/cassava (varies)"),
    (r"\bcà tím\b", "eggplant (cà tím; some avoid)"),
)

_NOT_JAIN_EN = _compile(NOT_JAIN_EN)
_JAIN_UNCERTAIN_EN = _compile(JAIN_UNCERTAIN_EN)
_NOT_JAIN_VI = _compile(NOT_JAIN_VI)
_JAIN_UNCERTAIN_VI = _compile(JAIN_UNCERTAIN_VI)


@dataclass(frozen=True)
class JainCheck:
    status: JainStatus
    matches: tuple[str, ...]

    @property
    def label(self) -> str:
        return {"not_jain": "Not Jain", "uncertain": "Uncertain", "jain": "Jain-friendly"}[self.status]


def classify_jain(original: str, english: str | None, veg: DietCheck) -> JainCheck:
    """Classify one ingredient for the Jain diet, building on its vegetarian check."""
    if veg.status == "non_veg":
        return JainCheck("not_jain", ("non-vegetarian",))
    texts = [original] + ([english] if english else [])
    folded = " | ".join(fold_text(t) for t in texts)
    vi = " | ".join(_vi_text(t) for t in texts)

    not_jain = _find(folded, vi, _NOT_JAIN_EN, _NOT_JAIN_VI)
    if not_jain:
        return JainCheck("not_jain", tuple(not_jain))
    uncertain = _find(folded, vi, _JAIN_UNCERTAIN_EN, _JAIN_UNCERTAIN_VI)
    if veg.status == "uncertain":
        uncertain = list(veg.matches) + uncertain
    if uncertain:
        return JainCheck("uncertain", tuple(uncertain))
    return JainCheck("jain", ())


def classify_jain_all(
    originals: Sequence[str], english: Sequence[str] | None, veg_checks: Sequence[DietCheck]
) -> list[JainCheck]:
    english = english if english and len(english) == len(originals) else [None] * len(originals)
    return [classify_jain(o, e, v) for o, e, v in zip(originals, english, veg_checks)]


def overall_jain(checks: Iterable[JainCheck]) -> JainStatus:
    statuses = {c.status for c in checks}
    if "not_jain" in statuses:
        return "not_jain"
    if "uncertain" in statuses:
        return "uncertain"
    return "jain"
