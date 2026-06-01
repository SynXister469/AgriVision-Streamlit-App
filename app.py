"""
Plant Disease Detector — Streamlit App
=======================================
Drag & drop a leaf photo → instant disease prediction using a TFLite model.

Files needed in the same folder as this script:
  • model.tflite   — your trained TFLite model
  • classes.txt    — one class name per line, in the same order as training
"""

import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from pathlib import Path
import tensorflow as tf

# ── Page config (must be first Streamlit call) ───────────────
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
)

# ── Styling ──────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a7a4a;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        font-size: 1.05rem;
        color: #555;
        margin: 0;
    }
    .result-healthy {
        background: #e8f5e9;
        border-left: 5px solid #2e7d32;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    .result-diseased {
        background: #fa9b05;
        border-left: 5px solid #e65100;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    .disease-name {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .crop-name {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 0;
    }
    .confidence-big {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a7a4a;
    }
    .info-card {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 0.5rem;
    }
    .info-card h4 {
        color: #333;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    .info-card p {
        color: #555;
        font-size: 0.9rem;
        margin: 0;
        line-height: 1.6;
    }
    .severity-high   { color: #c62828; font-weight: 700; }
    .severity-medium { color: #e65100; font-weight: 700; }
    .severity-low    { color: #2e7d32; font-weight: 700; }
    .upload-tip {
        text-align: center;
        color: #888;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #a5d6a7;
        border-radius: 12px;
        padding: 0.5rem;
        background: #f1f8f4;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────
MODEL_PATH   = Path("model.tflite")
CLASSES_PATH = Path("classes.txt")

# ── Disease knowledge base ───────────────────────────────────
# Contains description, organic & chemical treatment, and severity
# for every PlantVillage class. Severity: High / Medium / Low
DISEASE_DB = {
    # ── Pepper (Bell) ────────────────────────────────────────
    "Pepper__bell___Bacterial_spot": {
        "desc":     "Bacterial disease causing small water-soaked lesions on leaves that turn brown with yellow halos. Spreads rapidly in warm, wet conditions.",
        "organic":  "Use disease-free seeds. Apply copper-based bactericides every 7 days. Avoid overhead watering. Remove and destroy infected leaves.",
        "chemical": "Copper hydroxide or copper sulfate sprays at 7–10 day intervals. Acibenzolar-S-methyl as a plant defense activator.",
        "severity": "Medium",
    },
    "Pepper__bell___healthy": {
        "desc":     "No disease detected. Bell pepper plant is healthy and growing well.",
        "organic":  "Ensure consistent soil moisture. Mulch around base to retain water. Balanced fertilization with potassium.",
        "chemical": "No treatment needed.",
        "severity": "Low",
    },
    # ── Potato ───────────────────────────────────────────────
    "Potato___Early_blight": {
        "desc":     "Fungal disease (Alternaria solani) causing dark brown target-ring 'bullseye' lesions on older leaves first, working upward.",
        "organic":  "Remove and dispose of lower infected leaves. Apply neem oil or copper-based spray weekly. Avoid overhead watering.",
        "chemical": "Chlorothalonil, mancozeb, or azoxystrobin fungicides at 7–10 day intervals. Begin at first sign of symptoms.",
        "severity": "Medium",
    },
    "Potato___Late_blight": {
        "desc":     "Devastating oomycete disease (Phytophthora infestans) causing large, dark, water-soaked lesions with white mold on leaf underside. Can destroy a crop in days.",
        "organic":  "Remove and destroy infected plants immediately — do not compost. Apply copper-based fungicides preventively before symptoms appear.",
        "chemical": "Metalaxyl, chlorothalonil, or cymoxanil fungicides. Act within 24 hours — late blight spreads extremely fast in cool, wet weather.",
        "severity": "High",
    },
    "Potato___healthy": {
        "desc":     "No disease detected. Potato plant appears healthy.",
        "organic":  "Hill soil around stems as plant grows. Ensure consistent moisture. Rotate crops annually.",
        "chemical": "No treatment needed.",
        "severity": "Low",
    },
    # ── Tomato ───────────────────────────────────────────────
    "Tomato_Bacterial_spot": {
        "desc":     "Bacterial disease (Xanthomonas) causing small, water-soaked spots on leaves and fruit that turn dark brown with yellow halos.",
        "organic":  "Use copper-based bactericides. Avoid overhead irrigation. Use certified disease-free seeds. Remove infected debris.",
        "chemical": "Copper hydroxide or acibenzolar-S-methyl sprays every 5–7 days during wet weather.",
        "severity": "Medium",
    },
    "Tomato_Early_blight": {
        "desc":     "Fungal disease (Alternaria solani) causing dark concentric-ring bullseye lesions starting on lower, older leaves.",
        "organic":  "Remove infected lower leaves promptly. Apply copper or neem oil spray. Mulch soil to prevent splash spread.",
        "chemical": "Chlorothalonil, mancozeb, or azoxystrobin fungicides every 7 days.",
        "severity": "Medium",
    },
    "Tomato_Late_blight": {
        "desc":     "Highly destructive oomycete disease (Phytophthora infestans) causing large, dark, greasy-looking lesions. Spreads rapidly in cool, wet conditions.",
        "organic":  "Remove infected plants immediately. Copper-based sprays as prevention only — not curative.",
        "chemical": "Metalaxyl-M, cymoxanil, or fluopicolide fungicides. Treat immediately at first sign.",
        "severity": "High",
    },
    "Tomato_Leaf_Mold": {
        "desc":     "Fungal disease (Passalora fulva) causing pale yellow patches on upper leaf surface and olive-green velvety mold on the underside.",
        "organic":  "Reduce humidity below 85%. Improve greenhouse ventilation. Remove infected leaves. Avoid wetting foliage.",
        "chemical": "Chlorothalonil or mancozeb fungicides applied every 7 days.",
        "severity": "Medium",
    },
    "Tomato_Septoria_leaf_spot": {
        "desc":     "Fungal disease (Septoria lycopersici) causing numerous small, circular spots with dark borders and white centers, mainly on lower leaves.",
        "organic":  "Remove lower infected leaves. Mulch around base to reduce soil splash. Apply neem oil spray.",
        "chemical": "Chlorothalonil, copper, or mancozeb fungicides every 7–10 days starting when spots appear.",
        "severity": "Medium",
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "desc":     "Arachnid pest (Tetranychus urticae) causing tiny stippled yellow dots on leaves with fine webbing visible on the underside. Thrives in hot, dry conditions.",
        "organic":  "Spray forcefully with water to dislodge mites. Apply neem oil or insecticidal soap. Introduce predatory mites (Phytoseiidae).",
        "chemical": "Abamectin or bifenazate miticides. Rotate between chemical groups to prevent resistance.",
        "severity": "Medium",
    },
    "Tomato__Target_Spot": {
        "desc":     "Fungal disease (Corynespora cassiicola) producing brown, circular lesions with concentric target-like rings on leaves, stems, and fruit.",
        "organic":  "Remove infected tissue promptly. Apply copper-based sprays preventively in humid conditions.",
        "chemical": "Fluxapyroxad or azoxystrobin fungicides at first sign of disease.",
        "severity": "Medium",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "desc":     "Viral disease spread by whitefly (Bemisia tabaci) causing yellowing, upward curling, and stunting of young leaves. No cure once infected.",
        "organic":  "Control whitefly populations with yellow sticky traps and reflective mulch. Remove and destroy infected plants.",
        "chemical": "Imidacloprid or thiamethoxam insecticides to control the whitefly vector. No chemical cure for the virus itself.",
        "severity": "High",
    },
    "Tomato__Tomato_mosaic_virus": {
        "desc":     "Viral disease causing mosaic light-green and yellow mottling, leaf distortion, and stunted growth. Spreads through contact and infected tools.",
        "organic":  "Remove and destroy infected plants. Wash hands and tools with soap before handling plants. Use resistant varieties.",
        "chemical": "No chemical cure. Control aphid and whitefly vectors with insecticidal soap.",
        "severity": "High",
    },
    "Tomato_healthy": {
        "desc":     "No disease detected. Tomato plant is healthy and shows no visible symptoms.",
        "organic":  "Maintain consistent watering at soil level. Stake plants for good air circulation. Rotate crops yearly.",
        "chemical": "No treatment needed.",
        "severity": "Low",
    },
}


def get_disease_info(class_name: str) -> dict:
    """
    Return disease info using exact key match on your 15 classes.
    Falls back gracefully for any unexpected class names.
    """
    # Exact match (covers all 15 classes above)
    if class_name in DISEASE_DB:
        return DISEASE_DB[class_name]

    # Case-insensitive fallback
    for key, val in DISEASE_DB.items():
        if key.lower() == class_name.lower():
            return val

    # Generic fallback
    is_healthy = "healthy" in class_name.lower()
    return {
        "desc":     "Leaf appears healthy — no visible symptoms detected." if is_healthy
                    else "Disease detected based on visual symptoms in the leaf image.",
        "organic":  "Maintain regular monitoring and good agricultural practices." if is_healthy
                    else "Consult a local agricultural extension office for specific treatment advice.",
        "chemical": "No treatment needed." if is_healthy else "Consult local agricultural guidelines.",
        "severity": "Low" if is_healthy else "Medium",
    }


def parse_class_name(raw: str) -> tuple[str, str]:
    """
    Convert your class folder names into (Crop, Disease) human-readable labels.

    Handles your specific naming patterns:
      Pepper__bell___Bacterial_spot  → (Pepper Bell, Bacterial Spot)
      Tomato_Bacterial_spot          → (Tomato, Bacterial Spot)
      Tomato__Target_Spot            → (Tomato, Target Spot)
      Tomato__Tomato_YellowLeaf__Curl_Virus → (Tomato, Tomato Yellow Leaf Curl Virus)
      Tomato_healthy                 → (Tomato, Healthy)
    """
    import re

    # Detect crop: everything before the first single _ or __ or ___
    # For "Pepper__bell___Bacterial_spot" → crop = "Pepper bell"
    # For "Tomato_Bacterial_spot"         → crop = "Tomato"
    s = raw.strip()

    # Split on triple underscore first (separates crop from disease in Pepper classes)
    if "___" in s:
        parts   = s.split("___", 1)
        crop    = parts[0].replace("__", " ").replace("_", " ").replace(",", "").strip()
        disease = parts[1].replace("__", " ").replace("_", " ").strip()
    else:
        # Tomato classes: crop is always the first word before first underscore
        first_underscore = s.find("_")
        if first_underscore != -1:
            crop    = s[:first_underscore]
            disease = s[first_underscore:].lstrip("_").replace("__", " ").replace("_", " ").strip()
        else:
            crop, disease = s, s

    # Clean up multiple spaces, fix YellowLeaf → Yellow Leaf
    disease = re.sub(r"([a-z])([A-Z])", r"\1 \2", disease)   # camelCase split
    disease = re.sub(r"\s+", " ", disease).strip()
    crop    = re.sub(r"\s+", " ", crop).strip().title()

    # Capitalize disease name properly
    disease = disease.title()

    return crop, disease


# ── Cached loaders ───────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"model.tflite not found. Place it next to app.py.")
        st.stop()
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    return interpreter


@st.cache_resource(show_spinner="Loading class names…")
def load_classes() -> list[str]:
    if not CLASSES_PATH.exists():
        st.error("classes.txt not found. Place it next to app.py.")
        st.stop()
    lines = CLASSES_PATH.read_text().strip().splitlines()
    # Handle formats: "0: ClassName", "0 ClassName", or just "ClassName"
    classes = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip leading "N:" or "N " prefix if present
        if line[0].isdigit():
            parts = line.split(None, 1)
            if len(parts) == 2:
                line = parts[1].lstrip(":").strip()
        classes.append(line)
    return classes


def preprocess(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """
    Resize → RGB → float32 → normalize to [0, 1] → add batch dim.
    Must match exactly what was done during training.
    """
    img = image.convert("RGB").resize(size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(image: Image.Image, interpreter, class_names: list) -> list[tuple]:
    """
    Fully bug-proof inference for TF 2.16+ quantized TFLite models.

    The root cause: TF 2.16+ has a bug in _get_tensor_details() that crashes
    on block-quantized models. This affects get_input_details(),
    get_output_details(), AND get_tensor_details() -- all three are broken.

    The ONLY safe TFLite call for quantized models is get_tensor(idx) itself.

    Strategy:
      1. Get input index via internal _interpreter.Inputs() (bypasses the bug)
      2. Set tensor, invoke the model
      3. After invoke(), scan indices 0..199 using get_tensor(idx) directly --
         this call is safe. Find the tensor shaped (1, num_classes) = output.
    """
    n = len(class_names)

    # ── Step 1: Input index ──────────────────────────────────
    input_idx = 0  # safe default for single-input models
    try:
        input_idx = interpreter._interpreter.Inputs()[0]
    except Exception:
        pass

    # ── Step 2: Determine input size and preprocess ──────────
    # PlantVillage standard is 224x224. Try to confirm via internal API.
    h = w = 224
    try:
        # Peek at input tensor shape via set_tensor with a dummy array first
        # then read it back -- avoids _get_tensor_details entirely
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        interpreter.set_tensor(input_idx, dummy)
        # If it didn't raise, 224x224 is correct
    except Exception:
        # Try 256x256 as alternative
        try:
            dummy = np.zeros((1, 256, 256, 3), dtype=np.float32)
            interpreter.set_tensor(input_idx, dummy)
            h = w = 256
        except Exception:
            h = w = 224  # fall back

    input_data = preprocess(image, (w, h))
    interpreter.set_tensor(input_idx, input_data)

    # ── Step 3: Run inference ────────────────────────────────
    interpreter.invoke()

    # ── Step 4: Find output tensor by scanning with get_tensor() ──
    # get_tensor(idx) is safe -- it does NOT call _get_tensor_details.
    # After invoke(), the output tensor holds real probability values.
    output_idx = None

    # First try the internal Outputs() API
    try:
        output_idx = interpreter._interpreter.Outputs()[0]
        # Verify it looks right
        t = interpreter.get_tensor(output_idx)
        if not (t.ndim == 2 and t.shape[1] == n):
            output_idx = None  # wrong tensor, fall through to scan
    except Exception:
        output_idx = None

    # Fallback: brute-force scan using only get_tensor() (always safe)
    if output_idx is None:
        for idx in range(200):
            try:
                t = interpreter.get_tensor(idx)
                if t.ndim == 2 and t.shape[1] == n:
                    output_idx = idx
                    break
            except Exception:
                continue

    if output_idx is None:
        raise RuntimeError(
            f"Could not find output tensor matching {n} classes. "
            "Make sure classes.txt has exactly one class name per line, "
            "in the same order your model was trained on."
        )

    probs = interpreter.get_tensor(output_idx)[0]
    results = sorted(
        zip(class_names, [float(p) for p in probs]),
        key=lambda x: x[1],
        reverse=True,
    )
    return results


def confidence_color(conf: float) -> str:
    if conf >= 0.80:
        return "#2e7d32"
    elif conf >= 0.55:
        return "#e65100"
    else:
        return "#c62828"


# ── UI ───────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌿 Plant Disease Detector</h1>
    <p>Upload a leaf photo and get an instant AI-powered disease diagnosis</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Load model and classes on startup
interpreter  = load_model()
class_names  = load_classes()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About this app")
    st.info(
        "This app uses a deep learning model trained on the **PlantVillage** "
        "dataset to identify plant diseases from leaf photographs."
    )
    st.markdown("**Model details**")
    st.markdown(f"- Input size: `224 × 224 px`")
    st.markdown(f"- Classes: `{len(class_names)}`")
    st.markdown(f"- Format: `TFLite`")
    st.markdown("---")
    st.markdown("**Tips for best results**")
    st.markdown(
        "- 📸 Use good natural lighting\n"
        "- 🌿 Capture one leaf clearly\n"
        "- 🔍 Include the affected area\n"
        "- ❌ Avoid heavy shadows"
    )
    st.markdown("---")
    st.caption("Built with TensorFlow Lite + Streamlit")

# ── Main layout: two columns ─────────────────────────────────
col_upload, col_result = st.columns([1, 1.2], gap="large")

with col_upload:
    st.markdown("#### Upload Leaf Image")
    uploaded = st.file_uploader(
        label="Drop a leaf photo here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    st.markdown(
        '<p class="upload-tip">Supports JPG, PNG, WEBP · Works best with clear, well-lit photos</p>',
        unsafe_allow_html=True,
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_column_width=True, caption="Uploaded leaf")

with col_result:
    if not uploaded:
        st.markdown("#### Results will appear here")
        st.markdown(
            "<div style='text-align:center; padding:4rem 1rem; color:#aaa;'>"
            "<div style='font-size:4rem'>🌱</div>"
            "<p>Upload a leaf image to get started</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Analysing leaf…"):
            results = predict(image, interpreter, class_names)

        top_class, top_conf = results[0]
        crop, disease = parse_class_name(top_class)
        is_healthy    = "healthy" in top_class.lower()
        info          = get_disease_info(top_class)
        severity      = info["severity"]

        # ── Top prediction banner ────────────────────────────
        box_class = "result-healthy" if is_healthy else "result-diseased"
        icon      = "✅" if is_healthy else "⚠️"
        st.markdown(f"""
        <div class="{box_class}">
            <div class="disease-name">{icon} {disease}</div>
            <div class="crop-name">Crop: {crop}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence + severity ────────────────────────────
        m1, m2 = st.columns(2)
        with m1:
            conf_pct = round(top_conf * 100, 1)
            color    = confidence_color(top_conf)
            st.markdown(
                f"<div style='text-align:center; padding:0.8rem 0'>"
                f"<div style='font-size:0.85rem; color:#888; margin-bottom:4px'>Confidence</div>"
                f"<div class='confidence-big' style='color:{color}'>{conf_pct}%</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        with m2:
            sev_class = f"severity-{severity.lower()}"
            sev_label = {"High": "🔴 High", "Medium": "🟡 Medium", "Low": "🟢 Low"}[severity]
            st.markdown(
                f"<div style='text-align:center; padding:0.8rem 0'>"
                f"<div style='font-size:0.85rem; color:#888; margin-bottom:4px'>Severity</div>"
                f"<div style='font-size:1.3rem; font-weight:700' class='{sev_class}'>{sev_label}</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Disease description ──────────────────────────────
        st.markdown(
            f"<div class='info-card'>"
            f"<h4>📋 What is this?</h4>"
            f"<p>{info['desc']}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Treatment ────────────────────────────────────────
        if not is_healthy:
            t1, t2 = st.columns(2)
            with t1:
                st.markdown(
                    f"<div class='info-card'>"
                    f"<h4>🌱 Organic treatment</h4>"
                    f"<p>{info['organic']}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with t2:
                st.markdown(
                    f"<div class='info-card'>"
                    f"<h4>🧪 Chemical treatment</h4>"
                    f"<p>{info['chemical']}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

# ── Top-5 confidence chart ────────────────────────────────────
if uploaded:
    st.divider()
    st.markdown("#### Top 5 Predictions")

    top5        = results[:5]
    top5_labels = [parse_class_name(c)[1] + f" ({parse_class_name(c)[0]})" for c, _ in top5]
    top5_confs  = [round(conf * 100, 2) for _, conf in top5]
    bar_colors  = [
        "#2e7d32" if "healthy" in c.lower() else "#e65100"
        for c, _ in top5
    ]

    fig = go.Figure(go.Bar(
        x=top5_confs,
        y=top5_labels,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.1f}%" for v in top5_confs],
        textposition="outside",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis=dict(title="Confidence (%)", range=[0, max(top5_confs) * 1.25]),
        yaxis=dict(autorange="reversed"),
        height=240,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    st.plotly_chart(fig, use_container_width=True)

    # ── Low confidence warning ────────────────────────────────
    if top_conf < 0.55:
        st.warning(
            f"⚠️ Confidence is low ({conf_pct}%). "
            "Try retaking the photo with better lighting, closer to the leaf, "
            "and with the affected area clearly visible."
        )
