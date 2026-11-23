# 🏷️ Ingredient Scanner — mobile OCR app (Streamlit + EasyOCR)

A mobile-friendly web app that lets you **photograph the ingredients panel on
the back of a packaged product (or upload a photo) and extract the text**. Runs
on your PC and is used from your phone's browser over Wi-Fi — no app-store
install needed.

## Features

- Camera capture directly from the phone browser, or upload from the gallery
- OCR powered by [EasyOCR](https://github.com/JaidedAI/EasyOCR) (CPU, no system
  Tesseract install required)
- **English + Vietnamese on by default**; 17 languages total (Hindi, Marathi,
  Tamil, French, Arabic, Chinese, …). Invalid combinations are flagged before
  the model loads (EasyOCR allows only one non-Latin script, paired with English only)
- **Tuned for small label print**: grayscale + contrast stretch + sharpen
  preprocessing, and 1.5× upsampling before text detection (both toggleable)
- **Ingredient list parser**: finds the `Ingredients:` / `Thành phần:` header
  (bilingual headers like `THÀNH PHẦN / INGREDIENTS:` handled), stops at the
  next section (allergens / `Dị ứng`, storage / `Bảo quản`, nutrition, …),
  splits on top-level commas so `Chocolate (sugar, cocoa butter)` stays one
  item, and offers a `.csv` download
- **Translation to English**: the parsed ingredient list is translated with a
  free online service (Google Translate, automatically falling back to
  MyMemory if Google is unavailable/blocked). Items are sent together so the
  translator has context (`đường` → "sugar", not "road"). English is shown in
  bold with the original beside it; the `.csv` contains both columns
- **Vegetarian check**: every ingredient is classified 🟢 vegetarian /
  🟠 uncertain / 🔴 non-vegetarian using English *and* Vietnamese term lists
  (checked on both the original text and the translation). **Egg and egg
  derivatives count as non-vegetarian.** A green banner means all ingredients
  are vegetarian; a red banner lists the offending ingredients. Additives that
  are only *sometimes* animal-derived (E471, glycerin, rennet, …) are amber so
  the red verdict stays reliable. The `.csv` includes `diet` and `flagged_terms`
  columns
- Detected text regions drawn on the image
- Full text assembled in reading order, with one-tap copy and `.txt` download
- Manual 0/90/180/270° rotation plus optional auto-rotation detection
- Adjustable confidence threshold and paragraph grouping
- Automatic EXIF rotation fix and downscaling of large phone photos

## Quick start

```powershell
cd vs_android_streamlit
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The first OCR run downloads the model weights (~100 MB; English + Vietnamese
share the Latin recognition model) into `~/.EasyOCR`; subsequent runs are instant.

## Scanning tips for packaged products

- Flatten the panel and fill the frame with the ingredients block.
- Avoid glare from foil/glossy plastic — tilt the pack slightly away from the light.
- Hold the phone parallel to the label and tap to focus on the small print.
- If the photo is sideways, use **Rotate image** under the photo.

## Use it from your phone

### Option A — Streamlit Community Cloud (recommended: works anywhere, HTTPS so the camera works)

1. Push this folder to a GitHub repository (public, or private — Community
   Cloud supports both; the `.gitignore` already excludes `.venv` and secrets).
2. Go to <https://share.streamlit.io>, sign in with GitHub and click **Create app → Deploy a public app from GitHub**.
3. Pick the repo/branch, set **Main file path** to `app.py`.
4. Open **Advanced settings**:
   - **Python version: 3.12** (Community Cloud only supports non-EOL releases;
     the local 3.14 is not available there yet — the code runs on both).
   - **Secrets**: optionally paste `APP_PASSWORD = "your-password"` to put the
     app behind a login (see `.streamlit/secrets.example.toml`). Without it the
     app is open to anyone who has the URL.
5. Click **Deploy**. The first build installs CPU-only PyTorch (a few minutes);
   the first scan then downloads the EasyOCR weights (~100 MB). Later cold
   starts after the app sleeps repeat that download, so the first scan after a
   pause takes longer.
6. Open `https://<your-app>.streamlit.app` on your phone and **Add to Home
   Screen** for an app-like icon. The camera tab works because the site is HTTPS.

**How the app is kept inside Community Cloud's free limits** (690 MB–2.7 GB RAM):

| Measure | Where | Effect |
| --- | --- | --- |
| CPU-only torch wheels | `requirements.txt` (`--extra-index-url …/whl/cpu`) | avoids ~2.5 GB of CUDA libraries |
| Detector canvas capped at 1280 px | `ocr.py` → `DETECTOR_CANVAS` | peak RAM ≈ 1.3–1.5 GB instead of 4 GB; recognition still uses the full-res crop |
| One model in memory | `app.py` → `get_reader(max_entries=1)` | switching languages swaps the model instead of stacking |
| Serialised OCR | `app.py` → `ocr_lock()` | concurrent users queue rather than multiplying peak RAM |
| Bounded result cache | `extract_text(max_entries=20)` | old images don't accumulate |

Expect **~15–60 s per scan** on the shared Community Cloud CPU; it's faster on
a PC. On a machine with more RAM you can raise detection resolution with
`OCR_CANVAS_SIZE=1600` (≈2.3 GB peak) or `2560` (≈4 GB peak).

Community Cloud egress IPs are shared, so Google Translate often returns 429
there; the **Auto** translation setting falls back to MyMemory automatically.

### Option B — same Wi-Fi as your PC

1. Make sure the phone and PC are on the same Wi-Fi network.
2. Find the PC's IP: `ipconfig` → *IPv4 Address* (e.g. `192.168.1.20`).
3. Open `http://192.168.1.20:8501` on the phone.
4. Optional: "Add to Home Screen" in Chrome/Safari for an app-like icon.

> **Camera access over HTTP:** browsers only expose the camera on `localhost`
> or **HTTPS**. Over plain HTTP on a LAN IP the *Camera* tab may be blocked, but
> the *Upload* tab still works (on phones it offers "Take photo" as a source).
> For full camera support, either:
> - run behind an HTTPS tunnel such as `ngrok http 8501` or `cloudflared tunnel --url http://localhost:8501`, or
> - use Streamlit's built-in TLS by adding `sslCertFile`/`sslKeyFile` under
>   `[server]` in `.streamlit/config.toml`, or
> - on Android Chrome, add `http://<pc-ip>:8501` to
>   `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.

## Project layout

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit UI (camera / upload, settings, ingredient list, full text) |
| `ocr.py` | OCR helpers: label enhancement, EasyOCR wrapper, reading-order text assembly, ingredient parser, language validation |
| `translate.py` | Ingredient translation to English via `deep-translator` (Google → MyMemory fallback, context-preserving chunking) |
| `diet.py` | Vegetarian / non-vegetarian classifier (EN + VI term lists, egg = non-veg, uncertain additives) |
| `.streamlit/config.toml` | Server binds to `0.0.0.0` so phones can connect; theme |
| `.streamlit/secrets.example.toml` | Template for the optional `APP_PASSWORD` login |
| `tests/test_ocr.py` | Unit tests for the pure OCR helpers |
| `tests/test_translate.py` | Unit tests for translation chunking and backend fallback (no network) |
| `tests/test_diet.py` | Unit tests for the vegetarian classifier (egg, accent-sensitive Vietnamese, E-numbers, false-positive guards) |

> **Disclaimer:** the vegetarian check is keyword-based and only as good as the
> OCR, the translation and the term lists in `diet.py`. Treat a green result as
> "nothing suspicious found", not as certification — always double-check the
> physical label for allergies or strict dietary needs. Add terms to the lists
> in `diet.py` to extend coverage.

## Notes on translation

- Translation needs internet; the OCR itself is fully offline.
- Only the **parsed ingredient list** is sent to the translation service, never
  the photo.
- Google's free endpoint rate-limits and sometimes CAPTCHA-blocks shared/corporate
  networks (HTTP 429). The default **Auto** service then falls back to MyMemory
  (free, ~500 characters per request, generous daily quota). You can pin either
  service in *Settings → Translation service*, or turn translation off.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
