"""Streamlit mobile OCR app: scan the ingredients panel on the back of a packaged
product with the camera (or upload a photo) and extract the text with EasyOCR.
"""

from __future__ import annotations

import hmac
import io
import threading

import streamlit as st
from PIL import Image, UnidentifiedImageError
from streamlit.errors import StreamlitSecretNotFoundError

from diet import classify_ingredients, classify_jain_all, overall_jain, overall_status
from ocr import (
    DEFAULT_LANGUAGES,
    SUPPORTED_LANGUAGES,
    draw_boxes,
    extract_ingredients,
    normalise_languages,
    prepare_image,
    results_to_text,
    run_ocr,
    split_all_text,
    validate_languages,
)
from translate import (
    BACKEND_LABELS,
    BACKENDS,
    TranslationError,
    pick_source_language,
    translate_items,
)

st.set_page_config(
    page_title="Ingredient Scanner",
    page_icon="🏷️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Mobile-friendly tweaks: larger tap targets and full-width controls.
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 720px; }
      .stButton > button, .stDownloadButton > button { width: 100%; min-height: 3rem; font-size: 1.05rem; }
      textarea { font-size: 1rem !important; }
      [data-testid="stCameraInput"] button { min-height: 3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def require_password() -> None:
    """Gate the app behind a password when APP_PASSWORD is set in secrets.

    Without the secret (e.g. local development) the app is open.
    """
    try:
        expected = st.secrets.get("APP_PASSWORD")
    except (FileNotFoundError, StreamlitSecretNotFoundError):
        return
    if not expected or st.session_state.get("authenticated"):
        return
    st.title("🏷️ Ingredient Scanner")
    with st.form("login"):
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Open"):
            if hmac.compare_digest(password, str(expected)):
                st.session_state["authenticated"] = True
                st.rerun()
            st.error("Incorrect password.")
    st.stop()


require_password()


# Only one model is kept in memory (changing languages swaps it out), and OCR
# calls are serialised so simultaneous users can't multiply the peak memory.
@st.cache_resource(show_spinner="Loading OCR model (first run downloads weights)...", max_entries=1)
def get_reader(languages: tuple[str, ...]):
    import easyocr

    return easyocr.Reader(list(languages), gpu=False)


@st.cache_resource
def ocr_lock() -> threading.Lock:
    return threading.Lock()


@st.cache_data(show_spinner=False, max_entries=20)
def extract_text(
    image_bytes: bytes,
    languages: tuple[str, ...],
    min_confidence: float,
    paragraph: bool,
    rotation: int,
    auto_rotate: bool,
    enhance: bool,
    small_text: bool,
):
    """Cached OCR pass keyed on the image bytes and settings."""
    image = prepare_image(Image.open(io.BytesIO(image_bytes)), rotation=rotation, enhance=enhance)
    reader = get_reader(languages)
    with ocr_lock():
        results = run_ocr(
            reader,
            image,
            min_confidence=min_confidence,
            paragraph=paragraph,
            auto_rotate=auto_rotate,
            small_text=small_text,
        )
    return image, results


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def translate_cached(items: tuple[str, ...], source: str, backend: str, multiple_sources: bool):
    return translate_items(list(items), source, backend=backend, multiple_sources=multiple_sources)


def csv_escape(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def load_image_bytes(uploaded) -> bytes | None:
    if uploaded is None:
        return None
    data = uploaded.getvalue()
    try:
        Image.open(io.BytesIO(data)).verify()
    except (UnidentifiedImageError, OSError):
        st.error("That file doesn't look like a supported image. Try a JPG, PNG or WEBP.")
        return None
    return data


st.title("🏷️ Ingredient Scanner")
st.caption("Point your camera at the ingredients panel on the back of a package and extract the text.")

with st.expander("📋 Tips for a good scan"):
    st.markdown(
        "- Flatten the panel as much as possible and fill the frame with the ingredients block.\n"
        "- Avoid glare from glossy foil or plastic — tilt the pack slightly away from the light.\n"
        "- Hold the phone parallel to the label; tap to focus on the small print.\n"
        "- Rotate the photo below so the text reads left-to-right."
    )

with st.expander("Settings", expanded=False):
    selected = st.multiselect(
        "Languages",
        options=list(SUPPORTED_LANGUAGES),
        default=list(DEFAULT_LANGUAGES),
        format_func=lambda code: SUPPORTED_LANGUAGES[code],
        help="English is always included. Fewer languages = faster, more accurate OCR.",
    )
    enhance = st.toggle(
        "Enhance for small label print",
        value=True,
        help="Grayscale + contrast stretch + sharpen. Helps faded ink and coloured backgrounds.",
    )
    small_text = st.toggle(
        "Optimise detection for small text",
        value=True,
        help="Upsamples the image before detection so fine print is found. Slower.",
    )
    min_confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.3, 0.05)
    paragraph = st.toggle("Group text into paragraphs", value=False)
    auto_rotate = st.toggle(
        "Auto-detect rotated text",
        value=False,
        help="Tries each text region at 90°, 180° and 270° too. More robust for sideways photos, but ~4x slower.",
    )
    show_boxes = st.toggle("Show detected regions on the image", value=True)
    st.divider()
    translate_on = st.toggle(
        "Translate ingredients to English",
        value=True,
        help="Uses a free online translation service; the parsed ingredient list is sent over the internet.",
    )
    backend = st.selectbox(
        "Translation service",
        options=list(BACKENDS),
        format_func=lambda code: BACKEND_LABELS[code],
        disabled=not translate_on,
    )

languages = tuple(normalise_languages(selected))
language_error = validate_languages(languages)
if language_error:
    st.error(language_error)
    st.stop()

camera_tab, upload_tab = st.tabs(["📸 Camera", "🖼️ Upload"])
with camera_tab:
    camera_file = st.camera_input("Take a photo", label_visibility="collapsed")
with upload_tab:
    upload_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
        label_visibility="collapsed",
    )

image_bytes = load_image_bytes(camera_file) or load_image_bytes(upload_file)

if image_bytes is None:
    st.info("Take a photo or upload an image to get started.")
    st.stop()

rotation = st.segmented_control(
    "Rotate image",
    options=[0, 90, 180, 270],
    default=0,
    format_func=lambda deg: "No rotation" if deg == 0 else f"{deg}°",
    help="Turn the photo so the text reads left-to-right before scanning.",
)
rotation = rotation or 0

with st.spinner("Reading label..."):
    image, results = extract_text(
        image_bytes, languages, min_confidence, paragraph, rotation, auto_rotate, enhance, small_text
    )

if show_boxes and results:
    st.image(draw_boxes(image, results), use_container_width=True)
else:
    st.image(image, use_container_width=True)

if not results:
    st.warning(
        "No text found. Try better lighting, a closer shot, rotating the image, "
        "or lowering the minimum confidence."
    )
    st.stop()

text = results_to_text(results)
avg_conf = sum(r.confidence for r in results) / len(results)

if avg_conf < 0.7:
    st.info(
        "Low confidence — if the text looks wrong, the photo may be sideways. "
        "Use **Rotate image** above or enable **Auto-detect rotated text** in Settings."
    )

ingredients = extract_ingredients(text)
header_found = bool(ingredients)
if not ingredients:
    # No "Ingredients:" header was read — still run the checks over everything
    # that was detected so the verdict is never silently skipped.
    ingredients = split_all_text(text)
if ingredients:
    if header_found:
        st.subheader(f"🧾 Ingredients ({len(ingredients)})")
    else:
        st.subheader("🧾 Detected items")
        st.warning(
            "Couldn't find an **Ingredients:** / **Thành phần:** header in the scan, so every "
            "detected phrase was checked instead. Re-shoot with the header in frame for a cleaner list."
        )

    english: list[str] | None = None
    source_lang = pick_source_language(languages)
    if translate_on and source_lang is not None:
        with st.spinner("Translating to English..."):
            try:
                translation = translate_cached(
                    tuple(ingredients),
                    source_lang,
                    backend,
                    multiple_sources=len(languages) > 2,
                )
                english = list(translation.items)
                st.caption(f"Translated by {BACKEND_LABELS[translation.backend]}.")
            except TranslationError as exc:
                st.warning(
                    "Couldn't reach a translation service, showing the original text. "
                    + " · ".join(f"{BACKEND_LABELS[name]}: {msg}" for name, msg in exc.errors.items())
                )
    elif translate_on:
        st.caption("Only English is selected, so there's nothing to translate.")

    checks = classify_ingredients(ingredients, english)
    verdict = overall_status(checks)
    jain_checks = classify_jain_all(ingredients, english, checks)
    jain_verdict = overall_jain(jain_checks)
    shown = english or ingredients

    def names(selected_checks, want_status):
        return ", ".join(f"**{name}**" for name, c in zip(shown, selected_checks) if c.status == want_status)

    if verdict == "non_veg":
        headline = "🔴 Non-vegetarian"
    elif jain_verdict == "jain":
        headline = "🟢 Vegetarian **and** Jain-friendly"
    elif jain_verdict == "not_jain":
        headline = "🟢 Vegetarian — 🚫 but **not** Jain-friendly (veg only)"
    else:
        headline = "🟢 Vegetarian — ❔ Jain suitability uncertain"
    if verdict == "uncertain" and jain_verdict != "not_jain":
        headline = headline.replace("🟢 Vegetarian", "🟠 Probably vegetarian")
    st.markdown(f"### {headline}")

    veg_col, jain_col = st.columns(2)
    with veg_col:
        if verdict == "non_veg":
            st.error(f"🔴 **NON-VEGETARIAN**\n\nContains: {names(checks, 'non_veg')}")
        elif verdict == "uncertain":
            st.warning(f"🟠 **Probably vegetarian**\n\nMay be animal-derived: {names(checks, 'uncertain')}")
        else:
            st.success("🟢 **VEGETARIAN**\n\nNo meat, fish, seafood or egg found")
    with jain_col:
        if jain_verdict == "not_jain":
            reasons = names(jain_checks, "not_jain") if verdict != "non_veg" else "non-vegetarian ingredients"
            st.error(f"🚫 **NOT JAIN**\n\nContains: {reasons}")
        elif jain_verdict == "uncertain":
            st.warning(f"❔ **JAIN: UNCERTAIN**\n\nVaries by household: {names(jain_checks, 'uncertain')}")
        else:
            st.success("🙏 **JAIN-FRIENDLY**\n\nNo root vegetables, honey, alcohol, fungi or figs")

    def item_line(i: int, orig: str, en: str | None, check, jain_check) -> str:
        main = f"**{en}** <span style=\"color:gray\">· {orig}</span>" if en else orig
        note = f" <span style=\"color:#c62828\">({', '.join(check.matches)})</span>" if check.matches else ""
        # Only add a Jain tag for Jain-specific reasons; non-veg items are already red.
        if check.status == "veg" and jain_check.status == "not_jain":
            note += f" <span style=\"color:#6d4c41\">🚫 Jain: {', '.join(jain_check.matches)}</span>"
        elif check.status != "non_veg" and jain_check.status == "uncertain":
            jain_only = [m for m in jain_check.matches if m not in check.matches]
            if jain_only:
                note += f" <span style=\"color:#b26a00\">❔ Jain: {', '.join(jain_only)}</span>"
        return f"{i}. {check.emoji} {main}{note}"

    st.markdown(
        "\n".join(
            item_line(i, orig, en, check, jain_check)
            for i, (orig, en, check, jain_check) in enumerate(
                zip(ingredients, english or [None] * len(ingredients), checks, jain_checks), 1
            )
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        "🟢 vegetarian · 🟠 uncertain (additive that may be animal-derived) · 🔴 non-vegetarian (egg counts as non-veg) · "
        "🚫 Jain: root vegetable / honey / alcohol / fungus / fig · ❔ Jain: avoidance varies by household"
    )

    def csv_rows(name_cols):
        return "\n".join(
            ",".join(name_cols(orig, en))
            + f",{check.label},{csv_escape('; '.join(check.matches))}"
            + f",{jain_check.label},{csv_escape('; '.join(jain_check.matches))}"
            for orig, en, check, jain_check in zip(ingredients, english or [None] * len(ingredients), checks, jain_checks)
        )

    if english:
        st.code("\n".join(english), language=None)
        csv_data = "ingredient_english,ingredient_original,diet,flagged_terms,jain,jain_flagged_terms\n" + csv_rows(
            lambda orig, en: (csv_escape(en), csv_escape(orig))
        )
    else:
        csv_data = "ingredient,diet,flagged_terms,jain,jain_flagged_terms\n" + csv_rows(
            lambda orig, en: (csv_escape(orig),)
        )

    st.download_button(
        "⬇️ Download ingredients (.csv)",
        data=csv_data,
        file_name="ingredients.csv",
        mime="text/csv",
    )

st.subheader("Extracted text")
st.text_area("Extracted text", value=text, height=220, label_visibility="collapsed")
st.code(text, language=None)  # st.code renders a one-tap copy button.

col1, col2 = st.columns(2)
with col1:
    st.download_button("⬇️ Download .txt", data=text, file_name="scanned_text.txt", mime="text/plain")
with col2:
    st.metric("Regions / avg. confidence", f"{len(results)} / {avg_conf:.0%}")

with st.expander("Detected regions"):
    st.dataframe(
        [{"Text": r.text, "Confidence": f"{r.confidence:.0%}"} for r in results],
        use_container_width=True,
        hide_index=True,
    )
