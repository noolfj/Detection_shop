"""
Streamlit dashboard for AI shoplifting detection.
Uploads a video, runs VideoMAE + YOLOv8 person detection,
and displays annotated results with alerts.

Launch:
    streamlit run src/dashboard/video_detection_app.py
"""
import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import time
from pathlib import Path
from collections import deque
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from transformers import VideoMAEForVideoClassification
from torchvision import transforms
from PIL import Image
from src.pipeline.detector import PersonDetector


# ---------------------------------------------------------------------------
# Model loading (cached so it only runs once)
# ---------------------------------------------------------------------------
HUGGINGFACE_MODEL_ID = "Awais1718/videomae-base-finetuned-kinetics-finetuned-shoplifting-dataset-2"


@st.cache_resource
def load_videomae(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use local model if it exists, otherwise download from HuggingFace Hub
    has_local = (
        Path(model_path).is_dir()
        and (Path(model_path) / "config.json").is_file()
        and ((Path(model_path) / "model.safetensors").is_file() or (Path(model_path) / "pytorch_model.bin").is_file())
    )
    if has_local:
        source = model_path
        st.info(f"Loading local model from {model_path}")
    else:
        source = HUGGINGFACE_MODEL_ID
        st.info(f"Local model not found. Downloading from HuggingFace: {source} ...")
    model = VideoMAEForVideoClassification.from_pretrained(source)
    model.to(device).eval()
    return model, device


@st.cache_resource
def load_yolo():
    return PersonDetector(conf_threshold=0.5)


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def predict_clip(model, frames, device):
    """Run VideoMAE on a list of 16 preprocessed tensors."""
    tensor = torch.stack(frames).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(pixel_values=tensor).logits
        probs = torch.softmax(logits, dim=1)[0]
    return float(probs[1])  # shoplifting probability


def annotate_frame(frame, detections, alert, smoothed_prob):
    """Draw bounding boxes and status bar on an OpenCV frame."""
    h, w = frame.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
        if alert:
            color = (0, 0, 255)
            label = f"SHOPLIFTING {smoothed_prob*100:.0f}%"
        else:
            color = (0, 200, 0)
            label = "Person"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    status = "ALERT: SUSPICIOUS ACTIVITY" if alert else "Normal"
    bar_color = (0, 0, 255) if alert else (0, 200, 0)
    cv2.putText(frame, f"{status}  prob={smoothed_prob*100:.1f}%",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2)

    if alert:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)

    return frame


def deduplicate_alerts(alerts, min_gap_frames=60):
    """Merge alerts that are within min_gap_frames of each other."""
    if not alerts:
        return []
    deduped = [alerts[0]]
    for a in alerts[1:]:
        if a["frame"] - deduped[-1]["frame"] >= min_gap_frames:
            deduped.append(a)
        elif a["smoothed_prob"] > deduped[-1]["smoothed_prob"]:
            deduped[-1] = a
    return deduped


# ---------------------------------------------------------------------------
# Processing -- stores everything in session_state
# ---------------------------------------------------------------------------
def run_detection(video_path, fps, total_frames,
                  threshold, frame_stride, window_stride, smooth_window, focus_mode):
    """Process video and save all results to st.session_state."""

    model, device = load_videomae(str(ROOT / "models" / "videomae-shoplifting-best"))
    yolo = load_yolo()

    cap = cv2.VideoCapture(video_path)
    clip_length = 16
    buffer_size = clip_length * frame_stride
    frame_buffer = deque(maxlen=buffer_size)
    recent_probs = deque(maxlen=smooth_window)

    alerts = []
    annotated_frames = []  # list of (frame_number, jpeg_bytes)
    raw_count = 0
    last_smoothed = 0.0
    last_alert = False
    current_detections = []

    progress = st.progress(0, text="Processing...")
    status_text = st.empty()
    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        raw_count += 1

        # Run person detection periodically (every 2 frames) to track active people
        if raw_count % 2 == 0 or not current_detections:
            current_detections = yolo.detect(frame)

        # Smart Crop on Person if focus mode enabled
        if focus_mode.startswith("Person Focus") and current_detections:
            best_det = max(current_detections, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
            h, w = frame.shape[:2]
            bx1, by1, bx2, by2 = best_det["bbox"]
            pw = int((bx2 - bx1) * 0.25)
            ph = int((by2 - by1) * 0.25)
            cx1 = max(0, int(bx1 - pw))
            cy1 = max(0, int(by1 - ph))
            cx2 = min(w, int(bx2 + pw))
            cy2 = min(h, int(by2 + ph))
            target_img = frame[cy1:cy2, cx1:cx2]
        else:
            target_img = frame

        pil = Image.fromarray(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
        frame_buffer.append(TRANSFORM(pil))

        if len(frame_buffer) >= buffer_size and raw_count % window_stride == 0:
            buf = list(frame_buffer)
            sampled = [buf[i * frame_stride] for i in range(clip_length)]
            prob = predict_clip(model, sampled, device)
            recent_probs.append(prob)
            last_smoothed = float(np.mean(recent_probs))
            last_alert = last_smoothed >= threshold

        if last_alert and current_detections:
            alerts.append({
                "frame": raw_count,
                "time": raw_count / fps,
                "smoothed_prob": last_smoothed,
                "n_persons": len(current_detections),
            })

        # Store ~2 annotated frames per second as compressed JPEG bytes
        if raw_count % max(fps // 2, 1) == 0:
            annotated = annotate_frame(frame.copy(), current_detections,
                                       last_alert, last_smoothed)
            _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            annotated_frames.append((raw_count, jpeg.tobytes()))

        pct = raw_count / total_frames if total_frames else 0
        elapsed = time.time() - t0
        proc_fps = raw_count / elapsed if elapsed > 0 else 0
        progress.progress(min(pct, 1.0),
                          text=f"Frame {raw_count}/{total_frames}  "
                               f"({proc_fps:.1f} fps)")

    cap.release()
    elapsed = time.time() - t0
    progress.progress(1.0, text="Done!")
    status_text.success(
        f"Processed {raw_count} frames in {elapsed:.1f}s "
        f"({raw_count/elapsed:.1f} fps)"
    )

    # Persist results in session_state so they survive reruns
    unique_alerts = deduplicate_alerts(alerts, min_gap_frames=int(fps * 1.5))

    # Pre-compute alert -> closest stored frame index
    alert_frame_indices = []
    for a in unique_alerts:
        target = a["frame"]
        closest_idx = min(
            range(len(annotated_frames)),
            key=lambda j: abs(annotated_frames[j][0] - target),
        ) if annotated_frames else 0
        alert_frame_indices.append(closest_idx)

    st.session_state["results"] = {
        "annotated_frames": annotated_frames,
        "alerts": unique_alerts,
        "alert_frame_indices": alert_frame_indices,
        "total_frames": raw_count,
        "elapsed": elapsed,
        "fps": fps,
    }


def run_live_stream(source, threshold, frame_stride, window_stride, smooth_window, focus_mode):
    """Process live video stream (webcam or RTSP/HTTP IP camera) in real time."""
    model, device = load_videomae(str(ROOT / "models" / "videomae-shoplifting-best"))
    yolo = load_yolo()

    # Open camera (DirectShow for Windows webcam index, standard for URL)
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        st.error(f"❌ Не удалось подключиться к камере: {source}. Проверьте подключение камеры или правильность ссылки.")
        st.session_state["live_monitoring"] = False
        return

    # Minimize internal OpenCV buffer for real-time low latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    clip_length = 16
    buffer_size = clip_length * frame_stride
    frame_buffer = deque(maxlen=buffer_size)
    recent_probs = deque(maxlen=smooth_window)

    alert_box = st.empty()
    video_box = st.empty()
    stats_box = st.empty()

    raw_count = 0
    alert_count = 0
    last_smoothed = 0.0
    last_alert = False
    current_detections = []
    t0 = time.time()

    while cap.isOpened() and st.session_state.get("live_monitoring", False):
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Видеопоток прерван или камера недоступна.")
            break
        raw_count += 1

        # Periodic person detection (every 2 frames)
        if raw_count % 2 == 0 or not current_detections:
            current_detections = yolo.detect(frame)

        # Smart Crop on Person if enabled
        if focus_mode.startswith("Person Focus") and current_detections:
            best_det = max(current_detections, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
            h, w = frame.shape[:2]
            bx1, by1, bx2, by2 = best_det["bbox"]
            pw = int((bx2 - bx1) * 0.25)
            ph = int((by2 - by1) * 0.25)
            cx1 = max(0, int(bx1 - pw))
            cy1 = max(0, int(by1 - ph))
            cx2 = min(w, int(bx2 + pw))
            cy2 = min(h, int(by2 + ph))
            target_img = frame[cy1:cy2, cx1:cx2]
        else:
            target_img = frame

        pil = Image.fromarray(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
        frame_buffer.append(TRANSFORM(pil))

        if len(frame_buffer) >= buffer_size and raw_count % window_stride == 0:
            buf = list(frame_buffer)
            sampled = [buf[i * frame_stride] for i in range(clip_length)]
            prob = predict_clip(model, sampled, device)
            recent_probs.append(prob)
            last_smoothed = float(np.mean(recent_probs))
            last_alert = last_smoothed >= threshold
            if last_alert and current_detections:
                alert_count += 1

        annotated = annotate_frame(frame.copy(), current_detections, last_alert, last_smoothed)

        elapsed = time.time() - t0
        proc_fps = raw_count / elapsed if elapsed > 0 else 0

        # Update live UI elements
        if last_alert:
            alert_box.error(f"🚨 **ВНИМАНИЕ: ПОДОЗРЕНИЕ НА КРАЖУ!** Вероятность: **{last_smoothed*100:.1f}%** (Порог: {threshold*100:.0f}%)")
        else:
            alert_box.success(f"🟢 **Обстановка в норме**. Вероятность подозрительных действий: {last_smoothed*100:.1f}%")

        img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        video_box.image(img_rgb, channels="RGB", use_container_width=True)
        stats_box.caption(f"⏱️ Кадров: {raw_count} | Скорость: {proc_fps:.1f} FPS | Зафиксировано тревог: {alert_count} | Людей в кадре: {len(current_detections)}")

    cap.release()



# ---------------------------------------------------------------------------
# Display -- reads from session_state, no reprocessing on slider change
# ---------------------------------------------------------------------------
def decode_frame(jpeg_bytes):
    """Decode stored JPEG bytes to RGB numpy array."""
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show_results():
    """Render results from session_state."""
    res = st.session_state["results"]
    annotated_frames = res["annotated_frames"]
    unique_alerts = res["alerts"]
    alert_frame_indices = res["alert_frame_indices"]
    raw_count = res["total_frames"]
    elapsed = res["elapsed"]
    fps = res["fps"]

    st.markdown("---")

    # ---- Alert gallery: show each alert as a thumbnail you can click ----
    if unique_alerts:
        st.error(f"SHOPLIFTING DETECTED: {len(unique_alerts)} alert(s)")

        # Show all alerts as a grid with image previews
        cols_per_row = min(len(unique_alerts), 4)
        for row_start in range(0, len(unique_alerts), cols_per_row):
            row_alerts = unique_alerts[row_start:row_start + cols_per_row]
            row_indices = alert_frame_indices[row_start:row_start + cols_per_row]
            cols = st.columns(cols_per_row)

            for col_i, (a, frame_idx) in enumerate(zip(row_alerts, row_indices)):
                alert_num = row_start + col_i + 1
                _, jpeg_bytes = annotated_frames[frame_idx]
                img_rgb = decode_frame(jpeg_bytes)

                with cols[col_i]:
                    st.image(img_rgb, width=320,
                             caption=f"Alert #{alert_num} | t={a['time']:.1f}s | "
                                     f"{a['smoothed_prob']*100:.0f}% conf")
                    detail = (
                        f"Frame {a['frame']} | "
                        f"{a['n_persons']} person(s) | "
                        f"Confidence: {a['smoothed_prob']*100:.1f}%"
                    )
                    st.caption(detail)
                    if st.button(f"Inspect Alert #{alert_num}", key=f"jump_{alert_num}"):
                        st.session_state["viewer_idx"] = frame_idx
    else:
        st.success("No shoplifting detected in this video.")

    # ---- Frame browser (slider) ----
    if annotated_frames:
        st.markdown("---")
        st.subheader("Frame Browser")
        st.caption("Scroll through all stored frames, or click an alert above to jump there.")

        default_idx = st.session_state.get("viewer_idx", 0)
        # Clamp in case stored index is out of range
        default_idx = min(default_idx, len(annotated_frames) - 1)

        idx = st.slider(
            "Frame position",
            0,
            len(annotated_frames) - 1,
            default_idx,
            key="frame_slider",
        )
        st.session_state["viewer_idx"] = idx

        fnum, jpeg_bytes = annotated_frames[idx]
        img_rgb = decode_frame(jpeg_bytes)
        # Capped width so the image doesn't stretch across the full page
        st.image(img_rgb, caption=f"Frame {fnum}  (t={fnum/fps:.1f}s)",
                 width=720)

    # ---- Stats ----
    st.markdown("---")
    st.subheader("Detection Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total frames", raw_count)
    col2.metric("Alerts", len(unique_alerts))
    col3.metric("Processing speed", f"{raw_count/elapsed:.1f} fps")


# ---------------------------------------------------------------------------
# Sidebar with descriptive help text
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.header("Settings")

        focus_mode = st.radio(
            "Target Focus Mode",
            ["Person Focus (Smart Crop)", "Full Frame (Wide View)"],
            index=0,
            help="Person Focus automatically crops detected individuals and analyzes their hands and movement up-close. Recommended for CCTV footage."
        )

        threshold = st.slider(
            "Detection threshold", 0.10, 0.95, 0.35, 0.05,
            help="Minimum smoothed shoplifting probability to trigger an alert."
        )
        st.caption(
            "**0.35 (recommended):** Catches shoplifting actions while filtering innocent movement. "
            "Lower it (0.20-0.30) for maximum sensitivity; raise it (0.50+) to reduce alerts."
        )

        frame_stride = st.slider(
            "Frame stride", 2, 12, 3, 1,
            help="Spacing between the 16 frames sampled for each clip."
        )
        st.caption(
            "**3 (default):** Spans ~1.5-2.0s (at 25-30fps), ideal for detecting fast concealment or grabbing gestures."
        )

        window_stride = st.slider(
            "Window stride", 2, 16, 6, 1,
            help="Run inference every N raw frames."
        )
        st.caption(
            "**6 (default):** Runs model every 6 frames (~0.2-0.25s) for smooth real-time monitoring on GPU."
        )

        smooth_window = st.slider(
            "Temporal smoothing", 1, 5, 2, 1,
            help="Average the last N predictions before deciding."
        )
        st.caption(
            "**2 (default):** Averages last 2 predictions to eliminate single-frame glitches."
        )

        st.markdown("---")
        st.subheader("About this model")
        st.markdown(
            "**VideoMAE-Base** -- a Video Masked Autoencoder pretrained on "
            "Kinetics-400, fine-tuned on UCF-Crime shoplifting vs. normal footage.\n\n"
            "Combined with **YOLOv8** person localization, the system analyzes "
            "hand motion, body posture shifts, and item concealment up close.\n\n"
            "| Metric | Value |\n"
            "|--------|-------|\n"
            "| Mode | Smart Person Focus |\n"
            "| Current threshold | {:.2f} |\n\n"
            "**Hardware Acceleration:** CUDA GPU enabled.".format(threshold)
        )

        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        st.info(f"Running on: **{device_name}** 🚀")

    return threshold, frame_stride, window_stride, smooth_window, focus_mode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="AI CCTV Shoplifting Detection", layout="wide")
    st.title("🛡️ AI CCTV Retail Security — Детекция краж")

    threshold, frame_stride, window_stride, smooth_window, focus_mode = render_sidebar()

    source_type = st.radio(
        "Выберите режим работы / Источник видео:",
        ["📁 Загрузить видеофайл (MP4, AVI)", "📹 Веб-камера (USB / Real-time)", "🌐 IP-камера (RTSP / CCTV видеопоток)"],
        horizontal=True
    )

    if source_type == "📁 Загрузить видеофайл (MP4, AVI)":
        uploaded = st.file_uploader("Загрузите видеозапись для анализа",
                                    type=["mp4", "avi", "mov", "mkv"])
        if uploaded is None:
            st.session_state.pop("results", None)
            st.info("ℹ️ Загрузите видеофайл для анализа.")
            return

        # Save upload to temp file (only once per file)
        file_id = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("uploaded_file_id") != file_id:
            suffix = Path(uploaded.name).suffix
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.read())
            tmp.flush()
            st.session_state["video_path"] = tmp.name
            st.session_state["uploaded_file_id"] = file_id
            st.session_state.pop("results", None)

        video_path = st.session_state["video_path"]

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        duration = total_frames / fps if fps else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Длительность", f"{duration:.1f}с")
        col2.metric("FPS", fps)
        col3.metric("Всего кадров", total_frames)
        col4.metric("Разрешение", f"{width}x{height}")

        # ---- process or show results ----
        if "results" not in st.session_state:
            if st.button("🚀 Начать анализ видео", type="primary"):
                run_detection(video_path, fps, total_frames,
                              threshold, frame_stride, window_stride, smooth_window, focus_mode)
                st.rerun()
        else:
            show_results()
            if st.button("🔄 Анализировать снова"):
                st.session_state.pop("results", None)
                st.rerun()

    elif source_type == "📹 Веб-камера (USB / Real-time)":
        st.subheader("📹 Прямая трансляция с веб-камеры (Real-time)")
        st.write("Система анализирует видеопоток с подключенной камеры в реальном времени с помощью VideoMAE + YOLOv8.")

        cam_id = st.number_input(
            "Индекс камеры (0 — встроенная / основная веб-камера, 1 или 2 — внешняя USB камера):",
            min_value=0, max_value=10, value=0, step=1
        )

        col_start, col_stop = st.columns([1, 1])
        if col_start.button("▶️ Начать видеонаблюдение", type="primary"):
            st.session_state["live_monitoring"] = True
            st.rerun()

        if col_stop.button("⏹️ Остановить наблюдение", type="secondary"):
            st.session_state["live_monitoring"] = False
            st.info("Видеонаблюдение остановлено.")

        if st.session_state.get("live_monitoring", False):
            run_live_stream(int(cam_id), threshold, frame_stride, window_stride, smooth_window, focus_mode)

    elif source_type == "🌐 IP-камера (RTSP / CCTV видеопоток)":
        st.subheader("🌐 Мониторинг сетевой IP-камеры CCTV (RTSP поток)")
        st.write("Подключение к профессиональным камерам видеонаблюдения магазина по протоколу RTSP или HTTP.")

        rtsp_url = st.text_input(
            "RTSP или HTTP ссылка видеопотока с камеры:",
            value=st.session_state.get("last_rtsp_url", ""),
            placeholder="rtsp://admin:password@192.168.1.100:554/stream1",
            help="Введите URL потока с камеры видеонаблюдения или со смартфона (через приложение IP Webcam / DroidCam)"
        )
        st.caption("💡 Примеры форматов: `rtsp://admin:12345@192.168.1.50:554/ch01/0` или `http://192.168.1.80:8080/video` (с телефона по Wi-Fi)")

        col_start, col_stop = st.columns([1, 1])
        if col_start.button("▶️ Подключиться к камере", type="primary"):
            if not rtsp_url.strip():
                st.warning("⚠️ Пожалуйста, укажите RTSP или HTTP ссылку на видеопоток.")
            else:
                st.session_state["last_rtsp_url"] = rtsp_url.strip()
                st.session_state["live_monitoring"] = True
                st.rerun()

        if col_stop.button("⏹️ Отключиться", type="secondary"):
            st.session_state["live_monitoring"] = False
            st.info("Подключение отключено.")

        if st.session_state.get("live_monitoring", False) and rtsp_url.strip():
            run_live_stream(rtsp_url.strip(), threshold, frame_stride, window_stride, smooth_window, focus_mode)


if __name__ == "__main__":
    main()

