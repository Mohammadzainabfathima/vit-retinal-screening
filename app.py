"""Streamlit app: retinal fundus screening with the trained ViT-B/16 (RFMiD).

Run on your own computer:
    pip install streamlit torch torchvision timm pillow numpy pandas
    streamlit run app.py
Put these files in the same folder as app.py:
    vit_best.pt, app_config.json   (from Google Drive > rfmid_results)
Optional (shown in the "Model performance" tab): any .png files, e.g. evaluation_dashboard.png
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import timm
import torch
import torchvision.transforms as T
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "vit_best.pt")
CONFIG = os.path.join(HERE, "app_config.json")

st.set_page_config(page_title="Retinal screening (ViT)", page_icon="👁️", layout="wide")


def crop_and_resize(img, size=256):
    """Same preprocessing as used to build the training images (Day 2)."""
    img = img.convert("RGB")
    gray = np.asarray(img.convert("L"))
    mask = gray > 15
    if mask.mean() > 0.05:
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        img = img.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.BILINEAR)


@st.cache_resource
def load_model():
    cfg = json.load(open(CONFIG))
    model = timm.create_model(cfg["timm_name"], pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    tf = T.Compose([T.Resize((cfg["img_size"], cfg["img_size"])), T.ToTensor(),
                    T.Normalize(cfg["mean"], cfg["std"])])
    return model, tf, cfg, device


def predict(img):
    model, tf, cfg, device = load_model()
    small = crop_and_resize(img, 256)
    x = tf(small).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x).squeeze(1).float()).item()
    return prob, small


# ------------------------------------------------------------------ page
st.title("👁️ Retinal fundus screening with a Vision Transformer")
st.warning(
    "Student research prototype trained on RFMiD. **Not a medical device and not a diagnosis.** "
    "It only works on retinal fundus photographs (the picture of the back of the eye taken with a fundus camera), "
    "not on phone photos of an eye or face."
)

if not (os.path.exists(WEIGHTS) and os.path.exists(CONFIG)):
    st.error("Missing `vit_best.pt` and/or `app_config.json` next to app.py. Download them from "
             "Google Drive > rfmid_results and put them in this folder.")
    st.stop()

model, tf, cfg, device = load_model()
thr = cfg["threshold"]
tm = cfg["test_metrics"]

with st.sidebar:
    st.header("Model")
    st.write(f"**Architecture:** ViT-B/16 (`{cfg['timm_name']}`)")
    st.write(f"**Running on:** {device}")
    st.write(f"**Decision threshold:** {thr:.3f} (chosen on the validation set)")
    st.subheader("Test-set results")
    st.metric("AUC", f"{tm['auc']:.3f}")
    c1, c2 = st.columns(2)
    c1.metric("Sensitivity", f"{tm['sensitivity']:.3f}")
    c2.metric("Specificity", f"{tm['specificity']:.3f}")
    st.metric("Balanced accuracy", f"{tm['balanced_acc']:.3f}")

tab_try, tab_perf = st.tabs(["Try it", "Model performance"])

with tab_try:
    left, right = st.columns([1, 1])
    with left:
        up = st.file_uploader("Upload a retinal fundus image", type=["png", "jpg", "jpeg"])
        truth = st.radio("If you know the true label, select it (optional, for checking the model):",
                         ["Unknown", "Healthy", "Abnormal"], horizontal=True)
    if "log" not in st.session_state:
        st.session_state.log = []
    if up is not None:
        img = Image.open(up)
        prob, small = predict(img)
        abnormal = prob >= thr
        with left:
            st.image(img, caption="Your upload", use_container_width=True)
        with right:
            st.subheader("Result")
            if abnormal:
                st.error(f"Flagged as **ABNORMAL** (possible disease). Please see an eye doctor to check.")
            else:
                st.success("Flagged as **HEALTHY** (no abnormality detected by this model).")
            st.write(f"Probability of abnormal: **{prob:.1%}** (threshold {thr:.1%})")
            st.progress(min(max(prob, 0.0), 1.0))
            st.image(small, caption="What the model sees (cropped, squared, resized)", width=256)
            if truth != "Unknown":
                correct = (truth == "Abnormal") == abnormal
                (st.success if correct else st.error)("Matches your label ✔" if correct else "Does NOT match your label ✘")
            key = (up.name, round(prob, 6), truth)
            if not st.session_state.log or st.session_state.log[-1]["key"] != key:
                st.session_state.log.append({"key": key, "file": up.name, "prob_abnormal": round(prob, 4),
                                             "prediction": "abnormal" if abnormal else "healthy", "your label": truth})
    if st.session_state.log:
        st.subheader("This session")
        df = pd.DataFrame(st.session_state.log).drop(columns="key")
        st.dataframe(df, use_container_width=True)
        known = df[df["your label"] != "Unknown"]
        if len(known):
            acc = (known["prediction"].str.lower() == known["your label"].str.lower()).mean()
            st.write(f"Agreement with your labels: **{acc:.0%}** over {len(known)} image(s)")
        if st.button("Clear session"):
            st.session_state.log = []
            st.rerun()

with tab_perf:
    st.write("Test-set confusion counts: "
             f"TN={tm['tn']}, FP={tm['fp']}, FN={tm['fn']}, TP={tm['tp']}")
    pngs = sorted(glob.glob(os.path.join(HERE, "*.png")))
    if not pngs:
        st.info("Copy the .png result figures (evaluation_dashboard.png, vit_loss_curves.png, ...) "
                "into this folder to see them here.")
    for p in pngs:
        st.image(p, caption=os.path.basename(p), use_container_width=True)
