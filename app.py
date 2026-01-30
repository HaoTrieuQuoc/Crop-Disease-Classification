import streamlit as st
import torch
import numpy as np
from PIL import Image
import tifffile as tiff
import pandas as pd
import altair as alt

from model_definitions import MultiModalNet

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Multimodal Crop Disease Classification",
    layout="wide"
)

# ================= CSS =================
st.markdown("""
<style>
.block-container { padding-top: 2rem; }
.model-box {
    border-radius: 18px;
    padding: 1.5rem;
    background-color: #f9fafb;
    border: 1px solid #e5e7eb;
}
.model-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 1rem;
}
.result-box {
    background-color: #ecfeff;
    padding: 1rem;
    border-radius: 14px;
    margin-top: 0.8rem;
    margin-bottom: 1.2rem;
}
.pred-title {
    font-weight: 600;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ================= CONFIG =================
RESNET_PATH = r"D:\Chuyên ngành 7\DAT301m\Model\Models\resnet_model.pt"
VIT_PATH    = r"D:\Chuyên ngành 7\DAT301m\Model\Models\vit_multimodal_model.pt"

CLASSES = ["Health", "Other", "Rust"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 224
HS_IN_CH = 101
MS_IN_CH = 5

st.title("🌿 Multimodal Crop Disease Classification")

# ================= SESSION STATE =================
if "resnet_results" not in st.session_state:
    st.session_state.resnet_results = []

if "vit_results" not in st.session_state:
    st.session_state.vit_results = []

# ================= LOAD MODEL =================
@st.cache_resource
def load_model(path, backbone):
    ckpt = torch.load(path, map_location=DEVICE)

    model = MultiModalNet(
        backbone=backbone,
        hs_in_ch=HS_IN_CH,
        use_rgb=True,
        use_ms=True,
        use_hs=True,
        n_classes=3
    )

    state_dict = ckpt["model"] if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE).eval()
    return model

resnet_model = load_model(RESNET_PATH, "resnet")
vit_model    = load_model(VIT_PATH, "vit")

# ================= LOAD INPUT =================
def load_rgb(file):
    img = Image.open(file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr).unsqueeze(0)

def load_spectral(file, channels):
    img = tiff.imread(file).astype(np.float32)
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    img = img[:channels]
    return torch.tensor(img).unsqueeze(0)

# ================= PREDICT =================
def predict_single_modal(model, modal, file):
    with torch.no_grad():
        rgb = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        ms  = torch.zeros(1, MS_IN_CH, IMG_SIZE, IMG_SIZE).to(DEVICE)
        hs  = torch.zeros(1, HS_IN_CH, IMG_SIZE, IMG_SIZE).to(DEVICE)

        if modal == "RGB":
            rgb = load_rgb(file).to(DEVICE)
            mask = torch.tensor([[1, 0, 0]], dtype=torch.float32).to(DEVICE)
        elif modal == "MS":
            ms = load_spectral(file, MS_IN_CH).to(DEVICE)
            mask = torch.tensor([[0, 1, 0]], dtype=torch.float32).to(DEVICE)
        else:
            hs = load_spectral(file, HS_IN_CH).to(DEVICE)
            mask = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(DEVICE)

        out = model(rgb, ms, hs, mask)
        prob = torch.softmax(out, dim=1)[0].cpu().numpy()

        return {
            "modal": modal,
            "filename": file.name,
            "prediction": CLASSES[np.argmax(prob)],
            "prob": dict(zip(CLASSES, prob))
        }

# ================= PLOT =================
def plot_prob_bar(prob_dict):
    df = pd.DataFrame({
        "Class": list(prob_dict.keys()),
        "Probability": [v * 100 for v in prob_dict.values()]
    })

    bar = alt.Chart(df).mark_bar(
        cornerRadiusTopLeft=6,
        cornerRadiusTopRight=6
    ).encode(
        x=alt.X("Class:N", title="", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Probability:Q", title="Probability (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color(
            "Class:N",
            scale=alt.Scale(
                range=["#A7C7E7", "#B7E4C7", "#FFD6A5"]  # pastel
            ),
            legend=None
        )
    )

    text = bar.mark_text(
        align="center",
        baseline="bottom",
        dy=-5,
        fontSize=13,
        fontWeight="bold"
    ).encode(
        text=alt.Text("Probability:Q", format=".1f")
    )

    st.altair_chart((bar + text), use_container_width=True)

# ================= INPUT =================
st.subheader("📥 Upload Images")

col1, col2, col3 = st.columns(3)

with col1:
    rgb_files = st.file_uploader("RGB images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

with col2:
    ms_files = st.file_uploader("MS images", type=["tif", "tiff", "png", "jpg"], accept_multiple_files=True)

with col3:
    hs_files = st.file_uploader("HS images", type=["tif", "tiff"], accept_multiple_files=True)

# ================= INFERENCE =================
col_r, col_v = st.columns(2)

def run_model(model, model_name, store_key):
    if st.button(f"🔮 Predict with {model_name}"):
        results = []
        for f in rgb_files or []:
            results.append(predict_single_modal(model, "RGB", f))
        for f in ms_files or []:
            results.append(predict_single_modal(model, "MS", f))
        for f in hs_files or []:
            results.append(predict_single_modal(model, "HS", f))
        st.session_state[store_key] = results

def render_results(results):
    for r in results:
        st.markdown(
            f"""
            <div class="result-box">
                <div class="pred-title">{r['modal']} — {r['filename']}</div>
                🌱 <b>Prediction:</b> {r['prediction']}
            </div>
            """,
            unsafe_allow_html=True
        )
        plot_prob_bar(r["prob"])

with col_r:
    st.markdown("<div class='model-box'>", unsafe_allow_html=True)
    st.markdown("<div class='model-title'>🧠 ResNet Multimodal</div>", unsafe_allow_html=True)
    run_model(resnet_model, "ResNet", "resnet_results")
    render_results(st.session_state.resnet_results)
    st.markdown("</div>", unsafe_allow_html=True)

with col_v:
    st.markdown("<div class='model-box'>", unsafe_allow_html=True)
    st.markdown("<div class='model-title'>🤖 ViT Multimodal</div>", unsafe_allow_html=True)
    run_model(vit_model, "ViT", "vit_results")
    render_results(st.session_state.vit_results)
    st.markdown("</div>", unsafe_allow_html=True)
