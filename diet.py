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
    (r"\b(shrimps?|prawns?|crabs?|lobsters?|squid|cuttlefish|octopus|clams?|mussels?|scallops?|oysters?)\b", "seafood"),
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
