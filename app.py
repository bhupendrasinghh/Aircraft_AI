# app_yolo_model_custom.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import tempfile
import zipfile
import os
import cv2
import numpy as np
import GPUtil

# ---------- Streamlit UI setup ----------
st.set_page_config(page_title="YOLO Dashboard (Custom Model)", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0b1220; color: #e6eef8; }
.stSidebar { background-color: #0f1724; }
.stButton>button { background: linear-gradient(90deg,#1f77b4,#135E96); color: white; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("✈️ Airplane Defect Detection — Custom Model Loader")
st.write("Upload your trained `.pt` model to use your own dataset’s classes instead of COCO defaults.")

# ---------- Sidebar: Model selection ----------
st.sidebar.header("Model & Device")

uploaded_model = st.sidebar.file_uploader(
    "Upload your trained YOLO model (.pt)",
    type=["pt"],
    help="Upload your custom YOLOv8 or YOLOv9 checkpoint (e.g. best.pt)"
)

custom_model_path_input = st.sidebar.text_input(
    "Or enter model path (if already on server):",
    value="runs/detect/airplane_defect_yolov8_refined/weights/best.pt"
)

# Fallback only if nothing custom found
fallback_choice = st.sidebar.selectbox(
    "Fallback base (only used if your model fails to load):",
    ["yolov8n.pt", "yolov8l.pt"]
)

use_cuda = torch.cuda.is_available()
device_arg = 0 if use_cuda else "cpu"
device_name = "CUDA (GPU)" if use_cuda else "CPU"
st.sidebar.write("Using device:", device_name)

# GPU info
if use_cuda:
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        st.sidebar.metric("GPU", gpu.name)
        st.sidebar.metric("Utilization", f"{gpu.load*100:.1f}%")
        st.sidebar.metric("VRAM used", f"{gpu.memoryUsed:.0f}/{gpu.memoryTotal:.0f} MB")

# ---------- Helper ----------
def save_uploaded_model_to_temp(uploaded_file) -> str:
    """Save uploaded model bytes to a temporary .pt file and return its path."""
    tmpf = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    tmpf.write(uploaded_file.read())
    tmpf.flush()
    tmpf.close()
    return tmpf.name

# ---------- Load model (cached) ----------
@st.cache_resource
def load_yolo_model(model_path: str, fallback_path: str):
    """Try loading user model; if fails, fallback to default."""
    try:
        if model_path and os.path.exists(model_path):
            st.sidebar.success(f"✅ Using your custom model: {os.path.basename(model_path)}")
            model = YOLO(model_path)
        else:
            st.sidebar.warning(f"⚠️ Custom model not found, loading fallback: {fallback_path}")
            model = YOLO(fallback_path)

        # Move to device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load custom model: {e}")
        st.sidebar.info(f"Loading fallback model: {fallback_path}")
        model = YOLO(fallback_path)
        return model

# Determine model path priority: uploaded -> local -> fallback
if uploaded_model:
    model_path_for_load = save_uploaded_model_to_temp(uploaded_model)
elif os.path.exists(custom_model_path_input.strip()):
    model_path_for_load = custom_model_path_input.strip()
else:
    model_path_for_load = ""

# Load YOLO model
model = load_yolo_model(model_path_for_load, fallback_choice)

# ---------- Display loaded classes ----------
names = getattr(model, "names", None)
if names and isinstance(names, dict) and len(names) > 0:
    st.sidebar.success("✅ Custom class names loaded")
    st.sidebar.write(", ".join(names.values()))
else:
    st.sidebar.warning("⚠️ Could not detect custom class names — using COCO fallback classes.")

# ---------- Detection controls ----------
confidence = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
input_mode = st.sidebar.radio("Input type", ["Image", "Video", "ZIP (images)"])

def to_pil_rgb(img_src):
    if isinstance(img_src, Image.Image):
        return img_src.convert("RGB")
    if isinstance(img_src, np.ndarray):
        if img_src.ndim == 3 and img_src.shape[2] == 3:
            return Image.fromarray(img_src[..., ::-1])
        return Image.fromarray(img_src)
    return Image.open(img_src).convert("RGB")

# ---------- Inference functions ----------
def run_image_inference(uploaded_file):
    pil = to_pil_rgb(uploaded_file)
    res = model.predict(source=pil, conf=confidence, device=device_arg)
    annotated = res[0].plot()
    annotated = annotated[..., ::-1] if isinstance(annotated, np.ndarray) else to_pil_rgb(annotated)
    return Image.fromarray(annotated), res

def run_video_inference(uploaded_video_file):
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(uploaded_video_file.read())
    tmp.flush()
    tmp.close()

    cap = cv2.VideoCapture(tmp.name)
    if not cap.isOpened():
        st.error("Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(tempfile.gettempdir(), "yolo_out.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_slot = st.empty()
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)
    i = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        res = model.predict(source=frame_bgr, conf=confidence, device=device_arg)
        ann = res[0].plot()
        if ann is None:
            ann = frame_bgr
        out.write(ann)
        frame_rgb = ann[..., ::-1]
        frame_slot.image(frame_rgb, use_column_width=True)
        i += 1
        progress.progress(min(1.0, i / total))

    cap.release()
    out.release()
    st.success("Video processed successfully ✅")
    st.video(out_path)

def run_zip_inference(uploaded_zip):
    tmpdir = tempfile.mkdtemp()
    zip_path = os.path.join(tmpdir, "upload.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.read())
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmpdir)
    imgs = []
    for root, _, files in os.walk(tmpdir):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                imgs.append(os.path.join(root, fn))
    if not imgs:
        st.warning("No images found inside ZIP.")
        return
    for p in sorted(imgs):
        pil = to_pil_rgb(p)
        res = model.predict(source=pil, conf=confidence, device=device_arg)
        ann = res[0].plot()
        ann = ann[..., ::-1] if isinstance(ann, np.ndarray) else to_pil_rgb(ann)
        st.image(ann, caption=os.path.basename(p), use_column_width=True)

# ---------- Main UI ----------
st.markdown("### Input / Run")

if input_mode == "Image":
    up = st.file_uploader("Upload image (jpg/png)", type=["jpg", "jpeg", "png"])
    if up:
        original = to_pil_rgb(up)
        st.image(original, caption="Original", use_column_width=True)
        if st.button("Run Detection"):
            with st.spinner("Running inference..."):
                ann_img, res = run_image_inference(up)
                st.image(ann_img, caption="Detection Result", use_column_width=True)
                boxes = res[0].boxes
                if boxes is None or len(boxes) == 0:
                    st.info("No detections found.")
                else:
                    st.subheader("Detections:")
                    for b in boxes:
                        cls_i = int(b.cls[0])
                        conf = float(b.conf[0])
                        cls_name = model.names.get(cls_i, str(cls_i)) if hasattr(model, "names") else str(cls_i)
                        st.write(f"- {cls_name}: {conf:.3f}")

elif input_mode == "Video":
    upv = st.file_uploader("Upload video (mp4/mov/avi)", type=["mp4", "mov", "avi"])
    if upv and st.button("Run Video Detection"):
        run_video_inference(upv)

else:  # ZIP
    upzip = st.file_uploader("Upload ZIP with images", type=["zip"])
    if upzip and st.button("Run ZIP Detection"):
        run_zip_inference(upzip)

st.markdown("---")
st.markdown("🧠 **Tip:** Use your trained `best.pt` or `last.pt` from `runs/detect/.../weights/` to get your actual dataset classes.")
