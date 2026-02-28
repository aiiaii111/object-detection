import os
from io import BytesIO
from urllib.request import urlopen

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError


DEFAULT_IMAGE_URL = "https://learn.microsoft.com/azure/ai-services/computer-vision/media/quickstarts/presentation.png"
TEXTS = {
    "title": "Azure Vision Image Analyzer",
    "caption": "Shows Caption / Objects / Tags / OCR (READ) in one request.",
    "sidebar_input": "Input",
    "source": "Image Source",
    "url": "URL",
    "upload": "Upload",
    "ocr_conf": "OCR min confidence",
    "analyze": "Analyze",
    "image_url": "Image URL",
    "upload_image": "Upload image",
    "input_preview": "Input image",
    "press_analyze": "Set options and click `Analyze`.",
    "need_upload": "Please upload an image file first.",
    "need_url": "Please enter a valid image URL.",
    "missing_env": "Environment variables VISION_ENDPOINT / VISION_KEY are missing.",
    "analyzing": "Analyzing image...",
    "api_error": "Azure API error",
    "unexpected_error": "Unexpected error",
    "caption_fallback": "Caption is unavailable in this region, so only READ/OBJECTS/TAGS are shown.",
    "desc": "Description (Caption)",
    "desc_none": "No description detected.",
    "objects": "Objects",
    "objects_none": "No objects detected.",
    "objects_img": "Detected Objects + Caption",
    "tags": "Tags",
    "tags_none": "No tags detected.",
    "ocr": "OCR (READ)",
    "ocr_none": "No text detected above the confidence threshold.",
    "bbox_preview_failed": "Failed to render bounding-box image",
}


def get_client():
    endpoint = os.environ.get("VISION_ENDPOINT")
    key = os.environ.get("VISION_KEY")
    if not endpoint or not key:
        raise ValueError(
            "Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'. "
            "Set both before running Streamlit."
        )
    return ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))


def _analyze_once(client, visual_features, image_url=None, image_bytes=None, language="en", gender_neutral_caption=True):
    kwargs = {"visual_features": visual_features, "language": language}
    if gender_neutral_caption:
        kwargs["gender_neutral_caption"] = True
    if image_url:
        return client.analyze_from_url(image_url=image_url, **kwargs)
    return client.analyze(image_data=image_bytes, **kwargs)


def analyze_with_fallback(client, image_url=None, image_bytes=None, language="en"):
    base_kwargs = {
        "visual_features": [VisualFeatures.CAPTION, VisualFeatures.READ, VisualFeatures.OBJECTS, VisualFeatures.TAGS]
    }
    used_language = language
    language_fallback = False
    try:
        result = _analyze_once(
            client,
            visual_features=base_kwargs["visual_features"],
            image_url=image_url,
            image_bytes=image_bytes,
            language=used_language,
            gender_neutral_caption=True,
        )
        return result, False, used_language, language_fallback
    except HttpResponseError as exc:
        if "NotSupportedLanguage" in str(exc) and used_language != "en":
            used_language = "en"
            language_fallback = True
            result = _analyze_once(
                client,
                visual_features=base_kwargs["visual_features"],
                image_url=image_url,
                image_bytes=image_bytes,
                language=used_language,
                gender_neutral_caption=True,
            )
            return result, False, used_language, language_fallback
        if "feature 'Caption' is not supported in this region" not in str(exc):
            raise
        try:
            result = _analyze_once(
                client,
                visual_features=[VisualFeatures.READ, VisualFeatures.OBJECTS, VisualFeatures.TAGS],
                image_url=image_url,
                image_bytes=image_bytes,
                language=used_language,
                gender_neutral_caption=False,
            )
            return result, True, used_language, language_fallback
        except HttpResponseError as exc2:
            if "NotSupportedLanguage" in str(exc2) and used_language != "en":
                used_language = "en"
                language_fallback = True
                result = _analyze_once(
                    client,
                    visual_features=[VisualFeatures.READ, VisualFeatures.OBJECTS, VisualFeatures.TAGS],
                    image_url=image_url,
                    image_bytes=image_bytes,
                    language=used_language,
                    gender_neutral_caption=False,
                )
                return result, True, used_language, language_fallback
            raise


def extract_objects(result):
    rows = []
    if result.objects is None or not result.objects.list:
        return rows
    for detected_object in result.objects.list:
        if not detected_object.tags:
            continue
        tag = detected_object.tags[0]
        box = detected_object.bounding_box
        rows.append(
            {
                "object": tag.name,
                "confidence(%)": round(tag.confidence * 100, 2),
                "left": box.x,
                "top": box.y,
                "right": box.x + box.width,
                "bottom": box.y + box.height,
            }
        )
    return rows


def extract_tags(result):
    rows = []
    if result.tags is None or not result.tags.list:
        return rows
    for tag in result.tags.list:
        rows.append({"tag": tag.name, "confidence(%)": round(tag.confidence * 100, 2)})
    return rows


def extract_read_lines(result, min_confidence):
    lines = []
    if result.read is None or not result.read.blocks:
        return lines
    for block in result.read.blocks:
        for line in block.lines:
            filtered_words = [word for word in line.words if word.confidence >= min_confidence]
            if not filtered_words:
                continue
            lines.append(" ".join(word.text for word in filtered_words))
    return lines


def load_pil_image(image_url=None, image_bytes=None):
    if image_bytes is not None:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    if image_url:
        with urlopen(image_url) as response:
            return Image.open(BytesIO(response.read())).convert("RGB")
    return None


def draw_object_boxes(image, result):
    if image is None or result.objects is None or not result.objects.list:
        return image

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font_size = max(18, int(min(annotated.width, annotated.height) * 0.03))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    for detected_object in result.objects.list:
        if not detected_object.tags:
            continue
        tag = detected_object.tags[0]
        box = detected_object.bounding_box
        left = box.x
        top = box.y
        right = box.x + box.width
        bottom = box.y + box.height

        draw.rectangle([(left, top), (right, bottom)], outline="green", width=3)
        draw.text((left + 6, top + 6), f"{tag.name}", fill="white", font=font)

    return annotated


def draw_caption_text(image, result):
    if image is None or result.caption is None:
        return image

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    caption_text = f"Caption: {result.caption.text} ({result.caption.confidence * 100:.1f}%)"

    # Draw a simple top banner so text stays readable on bright images.
    banner_height = 28
    draw.rectangle([(0, 0), (annotated.width, banner_height)], fill="black")
    draw.text((8, 6), caption_text, fill="white")
    return annotated


def run_streamlit_app():
    st.set_page_config(page_title="Azure Vision Image Analyzer", layout="wide")
    t = TEXTS

    with st.sidebar:
        st.header(t["sidebar_input"])
        source_type = st.radio(t["source"], [t["url"], t["upload"]], horizontal=True)
        min_confidence = st.slider(t["ocr_conf"], min_value=0.0, max_value=1.0, value=0.6, step=0.05)
        run_button = st.button(t["analyze"], type="primary", use_container_width=True)

    st.title(t["title"])
    st.caption(t["caption"])

    image_url = DEFAULT_IMAGE_URL
    image_bytes = None

    if source_type == t["url"]:
        image_url = st.text_input(t["image_url"], value=DEFAULT_IMAGE_URL)
        if image_url:
            st.image(image_url, caption=t["input_preview"], use_column_width=True)
    else:
        uploaded_file = st.file_uploader(t["upload_image"], type=["png", "jpg", "jpeg", "bmp", "webp"])
        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            st.image(Image.open(uploaded_file), caption=t["input_preview"], use_column_width=True)
            image_url = None

    if not run_button:
        st.info(t["press_analyze"])
        return

    if source_type == t["upload"] and image_bytes is None:
        st.warning(t["need_upload"])
        return
    if source_type == t["url"] and not image_url:
        st.warning(t["need_url"])
        return

    try:
        client = get_client()
    except ValueError as exc:
        st.error(f"{t['missing_env']} ({exc})")
        st.code(
            "export VISION_ENDPOINT='https://<resource>.cognitiveservices.azure.com/'\n"
            "export VISION_KEY='<your-key>'",
            language="bash",
        )
        return

    with st.spinner(t["analyzing"]):
        try:
            analyze_language = "en"
            result, caption_fallback, used_language, language_fallback = analyze_with_fallback(
                client, image_url=image_url, image_bytes=image_bytes, language=analyze_language
            )
        except HttpResponseError as exc:
            st.error(f"{t['api_error']}: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"{t['unexpected_error']}: {exc}")
            return

    if caption_fallback:
        st.info(t["caption_fallback"])
    if language_fallback:
        st.info(f"Requested analysis language '{analyze_language}' is unsupported. Used '{used_language}' instead.")

    st.subheader(t["desc"])
    if result.caption is not None:
        st.write(f"**{result.caption.text}**")
        st.write(f"Confidence: `{result.caption.confidence * 100:.2f}%`")
    else:
        st.write(t["desc_none"])

    st.subheader(t["objects"])
    object_rows = extract_objects(result)
    if object_rows:
        try:
            source_image = load_pil_image(image_url=image_url, image_bytes=image_bytes)
            annotated_image = draw_object_boxes(source_image, result)
            annotated_image = draw_caption_text(annotated_image, result)
            if annotated_image is not None:
                st.image(annotated_image, caption=t["objects_img"], use_column_width=True)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"{t['bbox_preview_failed']}: {exc}")
        st.dataframe(object_rows, use_container_width=True)
    else:
        st.write(t["objects_none"])

    st.subheader(t["tags"])
    tag_rows = extract_tags(result)
    if tag_rows:
        st.dataframe(tag_rows, use_container_width=True)
    else:
        st.write(t["tags_none"])

    st.subheader(t["ocr"])
    read_lines = extract_read_lines(result, min_confidence=min_confidence)
    if read_lines:
        for idx, line in enumerate(read_lines, start=1):
            st.write(f"{idx}. {line}")
    else:
        st.write(t["ocr_none"])


if __name__ == "__main__":
    run_streamlit_app()
