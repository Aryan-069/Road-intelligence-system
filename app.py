from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Road Intelligence | AI Dashboard",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path("best.pt")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0b1120;
    color: #e5e7eb;
}

[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #273449;
}

[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

.hero {
    padding: 28px 30px;
    border: 1px solid #26364e;
    border-radius: 18px;
    background: linear-gradient(120deg, #16243a, #102e35);
    margin-bottom: 24px;
}

.eyebrow {
    color: #5eead4;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero h1 {
    color: #f8fafc;
    font-size: 32px;
    margin: 10px 0;
}

.hero p {
    color: #b7c4d8;
    font-size: 15px;
    margin-bottom: 0;
}

.section-title {
    color: #f1f5f9;
    font-size: 21px;
    font-weight: 700;
    margin: 24px 0 12px;
}

div[data-testid="stMetric"] {
    background: #141f31;
    border: 1px solid #29384e;
    padding: 18px;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #aab9ce;
}

div[data-testid="stMetricValue"] {
    color: #5eead4;
}

div[data-testid="stFileUploader"] {
    background: #131e30;
    border: 1px dashed #3b536f;
    padding: 16px;
    border-radius: 14px;
}

.stButton button {
    background: #0d9488;
    color: white;
    border: 0;
    border-radius: 9px;
    font-weight: 700;
}

.stButton button:hover {
    background: #0f766e;
    color: white;
    border: 0;
}

[data-testid="stDataFrame"] {
    border: 1px solid #29384e;
    border-radius: 10px;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


with st.sidebar:
    st.markdown("## 🛣️ ROAD AI")
    st.caption("INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Overview", "Damage Inspection", "About Project"],
    )

    st.divider()
    st.markdown("**MODEL STATUS**")
    st.success("Model file available" if MODEL_PATH.exists() else "Model file missing")
    st.caption("YOLO26n · 4 damage classes")
    st.caption("Local inference mode")


if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH.resolve()}")
    st.stop()

model = load_model(str(MODEL_PATH.resolve()))

st.markdown(
    """
<div class="hero">
    <div class="eyebrow">AI-POWERED ROAD MONITORING</div>
    <h1>Road Intelligence Dashboard</h1>
    <p>Detect road damage, inspect model predictions, and review
    damage categories through a single interface.</p>
</div>
""",
    unsafe_allow_html=True,
)


if page == "Overview":
    st.markdown(
        '<div class="section-title">System Overview</div>', unsafe_allow_html=True
    )

    a, b, c, d = st.columns(4)
    a.metric("Detection Model", "YOLO26n")
    b.metric("Damage Classes", "4")
    c.metric("Training Epochs", "15")
    d.metric("Inference", "On demand")

    st.markdown(
        '<div class="section-title">Inspection Workflow</div>', unsafe_allow_html=True
    )

    x, y, z = st.columns(3)

    with x:
        st.markdown("### 01 · Upload")
        st.write("Select a road image for inspection.")

    with y:
        st.markdown("### 02 · Detect")
        st.write("The trained model predicts visible road damage.")

    with z:
        st.markdown("### 03 · Review")
        st.write("Review bounding boxes, confidence, and damage counts.")

    st.info("Open **Damage Inspection** from the sidebar to analyse an image.")


elif page == "Damage Inspection":
    st.markdown(
        '<div class="section-title">New Road Inspection</div>', unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload road image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    confidence = st.slider(
        "Detection confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Analysing road image..."):
            result = model.predict(
                source=image,
                imgsz=640,
                conf=confidence,
                device=0,
                verbose=False,
            )[0]

        annotated = Image.fromarray(result.plot()[..., ::-1])

        left, right = st.columns(2)

        with left:
            st.markdown("**Original Image**")
            st.image(image, use_container_width=True)

        with right:
            st.markdown("**AI Detection Output**")
            st.image(annotated, use_container_width=True)

        names = model.names
        counts = {name: 0 for name in names.values()}
        rows = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                score = float(box.conf.item())
                name = names[class_id]
                counts[name] += 1

                rows.append(
                    {
                        "Damage Type": name.replace("_", " ").title(),
                        "Confidence": round(score * 100, 2),
                    }
                )

        st.markdown(
            '<div class="section-title">Inspection Summary</div>',
            unsafe_allow_html=True,
        )

        cols = st.columns(5)
        cols[0].metric("Total Detections", len(rows))

        for col, (name, count) in zip(cols[1:], counts.items()):
            col.metric(name.replace("_", " ").title(), count)

        if rows:
            st.markdown("**Detection Details**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.warning(
                "No damage was detected at this threshold. "
                "This does not guarantee the road is damage-free."
            )

        st.caption(
            "Predictions are estimates and should be reviewed by a human "
            "before maintenance decisions are made."
        )

    else:
        st.info("Upload an image to start a road inspection.")


else:
    st.markdown(
        '<div class="section-title">About This Project</div>', unsafe_allow_html=True
    )

    st.write(
        "This project uses a trained YOLO26n object-detection model "
        "to identify four categories of road damage."
    )

    st.markdown("**Supported damage categories**")
    st.write("- Longitudinal crack")
    st.write("- Transverse crack")
    st.write("- Alligator crack")
    st.write("- Pothole")

    st.markdown("**Current capabilities**")
    st.write("- Image-based damage detection")
    st.write("- Bounding-box visualisation")
    st.write("- Confidence scores and category counts")

    st.warning(
        "Severity scoring, GPS-based reporting, repair prioritisation, "
        "and live monitoring are not implemented in this version."
    )

st.divider()
st.caption("ROAD AI · Academic project prototype · AI predictions require review")
