import json
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import streamlit as st

from main import AgeDetectionSystem
from storage import list_sessions, save_session


st.set_page_config(page_title="Age Detection Pro Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #0f1116 0%, #151a23 100%); }
    .block-container { padding-top: 1.6rem; }
    h1, h2, h3, p, label, .stMarkdown { color: #e6e9ef !important; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Age Detection Pro Dashboard")
st.caption("Real-time age estimation with tracking, analytics, persistent history, and API-ready workflow.")


def create_detector(
    use_gpu: bool,
    conf_threshold: float,
    min_face: int,
    smooth_alpha: float,
    track_distance: float,
    weights_dir: Optional[str],
) -> AgeDetectionSystem:
    return AgeDetectionSystem(
        weights_dir=weights_dir or None,
        use_gpu=use_gpu,
        confidence_threshold=conf_threshold,
        min_face_size=min_face,
        smooth_alpha=smooth_alpha,
        track_distance=track_distance,
    )


def require_login() -> bool:
    expected = os.getenv("STREAMLIT_DASH_PASSWORD", "").strip()
    if not expected:
        return True
    if st.session_state.get("auth_ok"):
        return True
    st.warning("Authentication is enabled for this dashboard.")
    pwd = st.text_input("Dashboard password", type="password")
    if st.button("Login", type="primary"):
        if pwd == expected:
            st.session_state["auth_ok"] = True
            st.success("Login successful.")
            st.rerun()
        else:
            st.error("Invalid password.")
    return False


def process_stream(
    detector: AgeDetectionSystem,
    source: Union[int, str],
    max_frames: int,
    show_panel: bool,
    save_video: bool,
) -> Tuple[dict, Optional[bytes]]:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error(f"Could not open source: {source!r}")
        return detector.analytics.summary_dict(), None

    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    progress = st.progress(0.0)

    output_temp = None
    writer = None
    if save_video:
        output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_temp.close()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_size = (width + 320, height) if show_panel else (width, height)
        writer = cv2.VideoWriter(output_temp.name, fourcc, fps, out_size)

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        annotated, _ = detector.annotate_frame(frame)
        display = detector._draw_panel(annotated, fps=0) if show_panel else annotated
        frame_placeholder.image(display, channels="BGR", use_container_width=True)

        if writer is not None:
            writer.write(display)

        frame_count += 1
        if max_frames > 0:
            progress.progress(min(frame_count / max_frames, 1.0))
            if frame_count >= max_frames:
                break
        else:
            # Keep progress bar active for open-ended streams.
            progress.progress(min((frame_count % 200) / 200.0, 1.0))

        if frame_count % 5 == 0:
            stats_placeholder.json(detector.analytics.summary_dict())

    cap.release()
    if writer is not None:
        writer.release()
    progress.progress(1.0)

    video_bytes = None
    if output_temp is not None:
        video_bytes = Path(output_temp.name).read_bytes()
        Path(output_temp.name).unlink(missing_ok=True)

    summary = detector.analytics.summary_dict()
    return summary, video_bytes


def to_csv_text(summary: dict) -> str:
    return "metric,value\n" + "\n".join(f"{k},{json.dumps(v)}" for k, v in summary.items())


if not require_login():
    st.stop()


with st.sidebar:
    st.header("Settings")
    mode = st.radio("Input mode", ["Webcam", "Upload video", "Upload image"])
    weights_dir = st.text_input("Weights folder (optional)", value="")
    use_gpu = st.toggle("Use GPU (CUDA)", value=False)
    conf_threshold = st.slider("Face confidence threshold", 0.1, 0.95, 0.55, 0.01)
    min_face = st.slider("Minimum face size (px)", 20, 200, 70, 5)
    smooth_alpha = st.slider("Temporal smoothing", 0.05, 0.95, 0.30, 0.01)
    track_distance = st.slider("Track match distance", 30.0, 250.0, 90.0, 5.0)
    show_panel = st.toggle("Show analytics panel", value=True)
    st.divider()
    st.caption("Tip: set env `STREAMLIT_DASH_PASSWORD` to protect this dashboard.")

detector = create_detector(
    use_gpu=use_gpu,
    conf_threshold=conf_threshold,
    min_face=min_face,
    smooth_alpha=smooth_alpha,
    track_distance=track_distance,
    weights_dir=weights_dir.strip() or None,
)

if mode == "Webcam":
    col1, col2 = st.columns(2)
    camera_index = col1.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
    max_frames = col2.number_input("Max frames (0 = unlimited)", min_value=0, value=300, step=50)
    save_video = st.checkbox("Save processed video", value=False)
    if st.button("Start webcam analysis", type="primary"):
        with st.spinner("Running webcam analysis..."):
            summary, video_bytes = process_stream(
                detector=detector,
                source=int(camera_index),
                max_frames=int(max_frames),
                show_panel=show_panel,
                save_video=save_video,
            )
        session_id = save_session(summary, source_type="webcam", source_label=f"camera_{int(camera_index)}")
        st.success(f"Saved session #{session_id} to SQLite history.")
        st.subheader("Session Summary")
        st.json(summary)
        st.download_button(
            "Download summary JSON",
            data=json.dumps(summary, indent=2),
            file_name="session_summary.json",
            mime="application/json",
        )
        st.download_button(
            "Download summary CSV",
            data=to_csv_text(summary),
            file_name="session_summary.csv",
            mime="text/csv",
        )
        if video_bytes is not None:
            st.download_button(
                "Download processed video",
                data=video_bytes,
                file_name="processed_output.mp4",
                mime="video/mp4",
            )

elif mode == "Upload video":
    uploaded_video = st.file_uploader("Upload video file", type=["mp4", "avi", "mov", "mkv"])
    max_frames = st.number_input("Max frames (0 = full video)", min_value=0, value=0, step=100)
    save_video = st.checkbox("Keep processed video for download", value=True)
    if uploaded_video is not None and st.button("Analyze uploaded video", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_video.read())
            temp_video_path = temp_video.name
        with st.spinner("Analyzing video..."):
            summary, video_bytes = process_stream(
                detector=detector,
                source=temp_video_path,
                max_frames=int(max_frames),
                show_panel=show_panel,
                save_video=save_video,
            )
        Path(temp_video_path).unlink(missing_ok=True)
        session_id = save_session(summary, source_type="upload_video", source_label=uploaded_video.name)
        st.success(f"Saved session #{session_id} to SQLite history.")
        st.subheader("Session Summary")
        st.json(summary)
        st.download_button(
            "Download summary JSON",
            data=json.dumps(summary, indent=2),
            file_name="video_summary.json",
            mime="application/json",
        )
        st.download_button(
            "Download summary CSV",
            data=to_csv_text(summary),
            file_name="video_summary.csv",
            mime="text/csv",
        )
        if video_bytes is not None:
            st.download_button(
                "Download processed video",
                data=video_bytes,
                file_name="processed_video.mp4",
                mime="video/mp4",
            )

else:
    uploaded_image = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"])
    if uploaded_image is not None and st.button("Analyze image", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
            temp_image.write(uploaded_image.read())
            temp_image_path = temp_image.name
        frame = cv2.imread(temp_image_path)
        Path(temp_image_path).unlink(missing_ok=True)
        if frame is None:
            st.error("Could not read uploaded image.")
        else:
            annotated, detections = detector.annotate_frame(frame)
            output = detector._draw_panel(annotated, fps=0) if show_panel else annotated
            st.image(output, channels="BGR", caption="Processed image", use_container_width=True)
            summary = detector.analytics.summary_dict()
            session_id = save_session(summary, source_type="upload_image", source_label=uploaded_image.name)
            st.success(f"Saved session #{session_id} to SQLite history.")
            st.subheader("Detections")
            st.json(detections)
            st.subheader("Summary")
            st.json(summary)
            st.download_button(
                "Download summary JSON",
                data=json.dumps(summary, indent=2),
                file_name="image_summary.json",
                mime="application/json",
            )
            st.download_button(
                "Download summary CSV",
                data=to_csv_text(summary),
                file_name="image_summary.csv",
                mime="text/csv",
            )

st.divider()
st.subheader("Recent Session History (SQLite)")
history = list_sessions(limit=20)
if not history:
    st.info("No sessions saved yet.")
else:
    st.dataframe(
        [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "source_type": row["source_type"],
                "source_label": row["source_label"],
                "frames_processed": row["summary"].get("frames_processed", 0),
                "estimated_avg_age": row["summary"].get("estimated_avg_age", 0),
                "unique_people_tracks": row["summary"].get("unique_people_tracks", 0),
            }
            for row in history
        ],
        use_container_width=True,
    )
