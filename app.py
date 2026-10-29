from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO

import tempfile
import cv2
import av
from streamlit_webrtc import (
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

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

class RoadDamageProcessor(VideoProcessorBase):
    def __init__(self, model):
        self.model = model

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        results = self.model.predict(
            source=image,
            imgsz=480,
            conf=0.25,
            device="cpu",
            verbose=False,
        )

        annotated = results[0].plot()

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24",
        )

with st.sidebar:
    st.markdown("## 🛣️ ROAD AI")
    st.caption("INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Damage Inspection",
            "Video Inspection",
            "Live Camera",
            "About Project",
        ],
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
                conf=0.10,
                device="cpu",
                verbose=False,
            )[0]

            st.write("Detections:", len(result.boxes))

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    score = float(box.conf.item())
                    st.write(
                        f"Class: {model.names[class_id]} | " f"Confidence: {score:.3f}"
                    )

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

elif page == "Video Inspection":
    st.markdown(
        '<div class="section-title">Recorded Video Inspection</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Upload a road video to detect cracks and potholes "
        "using your trained YOLO26n model."
    )

    video_file = st.file_uploader(
        "Upload road video",
        type=["mp4", "avi", "mov", "mkv"],
        key="road_video_upload",
    )

    if video_file is not None:
        if st.button("Analyze Video", type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
                temp_input.write(video_file.getvalue())
                input_path = temp_input.name

            output_path = tempfile.mktemp(suffix=".mp4")

            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if not cap.isOpened() or total_frames <= 0:
                st.error("Could not read this video. Try another file.")
            else:
                fps = fps if fps > 0 else 25
                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )

                progress = st.progress(0)
                status = st.empty()
                frame_number = 0
                detection_count = 0

                try:
                    while True:
                        success, frame = cap.read()
                        if not success:
                            break

                        results = model.predict(
                            source=frame,
                            imgsz=480,
                            conf=0.25,
                            device="cpu",
                            verbose=False,
                        )

                        annotated_frame = results[0].plot()
                        writer.write(annotated_frame)
                        detection_count += len(results[0].boxes)
                        frame_number += 1

                        if frame_number % 5 == 0:
                            progress.progress(min(frame_number / total_frames, 1.0))
                            status.write(
                                f"Processing frame {frame_number} "
                                f"of {total_frames}..."
                            )
                finally:
                    cap.release()
                    writer.release()

                progress.progress(1.0)
                status.success("Video processing complete!")

                st.metric("Frames processed", frame_number)
                st.metric("Total detections across frames", detection_count)

                with open(output_path, "rb") as output_file:
                    processed_video = output_file.read()

                st.video(processed_video)
                st.download_button(
                    "Download Annotated Video",
                    data=processed_video,
                    file_name="road_damage_annotated.mp4",
                    mime="video/mp4",
                )

                Path(input_path).unlink(missing_ok=True)
                Path(output_path).unlink(missing_ok=True)

elif page == "Live Camera":
    st.markdown(
        '<div class="section-title">Live Road Inspection</div>',
        unsafe_allow_html=True,
    )

    st.write("Allow camera access to detect road damage in real time.")

    webrtc_streamer(
        key="road-damage-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: RoadDamageProcessor(model),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.caption(
        "Predictions are AI estimates and may miss or misclassify damage."
    )

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
