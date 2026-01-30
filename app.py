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

st.title("Multimodal Crop Disease Classification")

# ================= SESSION STATE =================
if "resnet_result" not in st.session_state:
    st.session_state["resnet_result"] = None
if "vit_result" not in st.session_state:
    st.session_state["vit_result"] = None

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
    # Fix: Reset pointer to beginning before reading
    file.seek(0)
    img = Image.open(file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr).unsqueeze(0)

def load_spectral(file, channels):
    # Fix: Reset pointer to beginning before reading
    file.seek(0)
    img = tiff.imread(file).astype(np.float32)
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    img = img[:channels]
    return torch.tensor(img).unsqueeze(0)

# ================= PREDICT =================
def predict_multimodal(model, rgb_file, ms_file, hs_file):
    with torch.no_grad():
        # 1. Initialize empty tensors (zeros)
        rgb = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        ms  = torch.zeros(1, MS_IN_CH, IMG_SIZE, IMG_SIZE).to(DEVICE)
        hs  = torch.zeros(1, HS_IN_CH, IMG_SIZE, IMG_SIZE).to(DEVICE)
        
        # Initialize mask [RGB, MS, HS] to all zeros
        mask_val = [0, 0, 0]
        modalities_used = []

        # 2. Load inputs if present
        if rgb_file is not None:
            rgb = load_rgb(rgb_file).to(DEVICE)
            mask_val[0] = 1
            modalities_used.append("RGB")
        
        if ms_file is not None:
            ms = load_spectral(ms_file, MS_IN_CH).to(DEVICE)
            mask_val[1] = 1
            modalities_used.append("MS")
            
        if hs_file is not None:
            hs = load_spectral(hs_file, HS_IN_CH).to(DEVICE)
            mask_val[2] = 1
            modalities_used.append("HS")

        # Create mask tensor
        mask = torch.tensor([mask_val], dtype=torch.float32).to(DEVICE)

        # 3. Model Inference
        out = model(rgb, ms, hs, mask)
        prob = torch.softmax(out, dim=1)[0].cpu().numpy()

        return {
            "modalities": "+".join(modalities_used),
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
st.subheader("Upload Images (Single Sample)")

col1, col2, col3 = st.columns(3)

with col1:
    rgb_file = st.file_uploader("RGB Image", type=["jpg", "png", "jpeg"], accept_multiple_files=False)

with col2:
    ms_file = st.file_uploader("MS Image", type=["tif", "tiff", "png", "jpg"], accept_multiple_files=False)

with col3:
    hs_file = st.file_uploader("HS Image", type=["tif", "tiff"], accept_multiple_files=False)

# ================= INFERENCE =================
col_r, col_v = st.columns(2)

# --- ResNet Prediction ---
with col_r:
    st.markdown("<div class='model-box'>", unsafe_allow_html=True)
    st.markdown("<div class='model-title'>ResNet Multimodal</div>", unsafe_allow_html=True)
    
    if st.button("Predict ResNet", type="primary"):
        if not (rgb_file or ms_file or hs_file):
            st.warning("Please upload at least one image.")
        else:
            with st.spinner("Processing ResNet..."):
                st.session_state["resnet_result"] = predict_multimodal(resnet_model, rgb_file, ms_file, hs_file)
            
    if st.session_state["resnet_result"] is not None:
        res_resnet = st.session_state["resnet_result"]
        st.markdown(
            f"""
            <div class="result-box">
                <div class="pred-title">Input: {res_resnet['modalities']}</div>
                <b>Prediction:</b> {res_resnet['prediction']}
            </div>
            """,
            unsafe_allow_html=True
        )
        plot_prob_bar(res_resnet["prob"])

    st.markdown("</div>", unsafe_allow_html=True)

# --- ViT Prediction ---
with col_v:
    st.markdown("<div class='model-box'>", unsafe_allow_html=True)
    st.markdown("<div class='model-title'>ViT Multimodal</div>", unsafe_allow_html=True)
    
    if st.button("Predict ViT", type="primary"):
        if not (rgb_file or ms_file or hs_file):
            st.warning("Please upload at least one image.")
        else:
            with st.spinner("Processing ViT..."):
                st.session_state["vit_result"] = predict_multimodal(vit_model, rgb_file, ms_file, hs_file)
            
    if st.session_state["vit_result"] is not None:
        res_vit = st.session_state["vit_result"]
        st.markdown(
            f"""
            <div class="result-box">
                <div class="pred-title">Input: {res_vit['modalities']}</div>
                <b>Prediction:</b> {res_vit['prediction']}
            </div>
            """,
            unsafe_allow_html=True
        )
        plot_prob_bar(res_vit["prob"])
    st.markdown("</div>", unsafe_allow_html=True)
