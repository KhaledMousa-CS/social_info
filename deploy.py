import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import base64
from io import BytesIO

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MRI Scan · Brain Tumor Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">

<style>
:root {
  --bg:        #030b14;
  --bg2:       #071525;
  --surface:   #0c2036;
  --glass:     rgba(12,32,54,0.72);
  --border:    rgba(0,210,255,0.14);
  --cyan:      #00d2ff;
  --cyan-dim:  rgba(0,210,255,0.15);
  --teal:      #0af5c8;
  --red:       #ff4d6d;
  --red-dim:   rgba(255,77,109,0.15);
  --green:     #00f5a0;
  --green-dim: rgba(0,245,160,0.12);
  --muted:     #5a8aaa;
  --text:      #d4eaf7;
  --font-head: 'Syne', sans-serif;
  --font-mono: 'DM Mono', monospace;
}

html, body, [class*="css"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font-mono) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1200px; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: var(--cyan-dim); border-radius: 99px; }

/* ── FORCE SIDEBAR ALWAYS OPEN ─────────────── */
section[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
  min-width: 240px !important;
  max-width: 240px !important;
  transform: translateX(0) !important;
  visibility: visible !important;
}
/* Hide every known sidebar toggle class across Streamlit versions */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[aria-label="Close sidebar"],
button[aria-label="Open sidebar"],
.st-emotion-cache-1wqrzgl,
.st-emotion-cache-czk5ss,
.st-emotion-cache-1cypcdb {
  display: none !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

div[role="radiogroup"] label {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 9px 13px !important;
  margin-bottom: 7px !important;
  transition: all 0.2s ease !important;
  cursor: pointer !important;
  display: block !important;
  font-size: 13px !important;
}
div[role="radiogroup"] label:hover {
  border-color: var(--cyan) !important;
  background: var(--cyan-dim) !important;
}

[data-testid="stFileUploader"] {
  background: var(--surface) !important;
  border: 1.5px dashed rgba(0,210,255,0.22) !important;
  border-radius: 14px !important;
  padding: 1.2rem !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--cyan) !important; }

div[data-testid="stProgress"] > div > div {
  background: linear-gradient(90deg, var(--teal), var(--cyan)) !important;
  border-radius: 99px !important;
}
div[data-testid="stProgress"] {
  background: rgba(255,255,255,0.06) !important;
  border-radius: 99px !important;
}
.stSpinner > div { border-top-color: var(--cyan) !important; }

/* ── ANIMATIONS ────────────────────────────── */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(18px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes scanLine {
  0%   { top: 0%;  opacity: 1;   }
  90%  { top: 96%; opacity: 0.7; }
  100% { top: 0%;  opacity: 0;   }
}
@keyframes pulse-ring {
  0%   { box-shadow: 0 0 0 0    rgba(0,210,255,0.35); }
  70%  { box-shadow: 0 0 0 12px rgba(0,210,255,0);    }
  100% { box-shadow: 0 0 0 0    rgba(0,210,255,0);     }
}
@keyframes glow-text {
  0%, 100% { text-shadow: 0 0 10px var(--cyan); }
  50%       { text-shadow: 0 0 28px var(--cyan), 0 0 55px var(--teal); }
}
@keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0; } }

/* ── HERO ──────────────────────────────────── */
.hero { animation: fadeUp 0.65s ease both; padding: 1.2rem 0 0.8rem; }
.hero-badge {
  display: inline-flex; align-items: center; gap: 8px;
  font-size: 10px; font-weight: 500; letter-spacing: 2px; text-transform: uppercase;
  color: var(--cyan); background: var(--cyan-dim);
  border: 1px solid var(--border); border-radius: 99px;
  padding: 5px 13px; margin-bottom: 12px;
}
.hero-badge-dot {
  width: 5px; height: 5px; border-radius: 50%;
  background: var(--cyan); animation: blink 1.4s infinite;
}
.hero-title {
  font-family: var(--font-head) !important;
  font-size: clamp(1.9rem, 4vw, 2.9rem);
  font-weight: 800; line-height: 1.1; letter-spacing: -1px; margin: 0 0 8px;
  background: linear-gradient(130deg, #ffffff 0%, var(--cyan) 55%, var(--teal) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  animation: glow-text 4s ease-in-out infinite;
}
.hero-sub { font-size: 13px; color: var(--muted); line-height: 1.65; }
.hero-divider {
  height: 1px; background: linear-gradient(90deg, var(--cyan), transparent);
  opacity: 0.25; margin: 18px 0;
}

/* ── RESULT CARD ───────────────────────────── */
.result-card {
  background: var(--glass);
  backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--border);
  border-radius: 18px; padding: 18px; margin-bottom: 22px;
  animation: fadeUp 0.5s ease both;
  position: relative; overflow: hidden;
}
.result-card::before {
  content: ''; position: absolute; inset: 0;
  background: linear-gradient(135deg, rgba(0,210,255,0.04) 0%, transparent 55%);
  pointer-events: none;
}

/* ── SCAN IMAGE ────────────────────────────── */
.img-wrapper {
  position: relative; border-radius: 11px; overflow: hidden;
  border: 1px solid var(--border); margin-bottom: 13px;
}
.img-wrapper img { width: 100%; display: block; border-radius: 10px; }
.img-wrapper::after {
  content: ''; position: absolute; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--cyan), transparent);
  animation: scanLine 2.5s linear infinite;
}

/* ── DX TEXT ───────────────────────────────── */
.dx-label {
  font-family: var(--font-head) !important;
  font-size: 19px; font-weight: 700; margin: 0 0 3px;
}
.dx-label.tumor { color: var(--red); }
.dx-label.clear { color: var(--green); }
.dx-conf {
  font-size: 11px; color: var(--muted);
  letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 9px;
}

/* ── CONFIDENCE BAR ────────────────────────── */
.conf-track {
  height: 7px; border-radius: 99px;
  background: rgba(255,255,255,0.07); overflow: hidden; margin-bottom: 5px;
}
.conf-fill { height: 100%; border-radius: 99px; }
.conf-fill.ok     { background: linear-gradient(90deg, var(--teal), var(--cyan)); }
.conf-fill.danger { background: linear-gradient(90deg, var(--red), #ff8c5a); }

/* ── STATUS BADGE ──────────────────────────── */
.status-badge {
  display: inline-flex; align-items: center; gap: 8px;
  font-size: 12px; padding: 7px 14px; border-radius: 99px;
  margin-top: 9px; margin-bottom: 4px;
}
.status-badge.ok {
  background: var(--green-dim); border: 1px solid rgba(0,245,160,0.3);
  color: var(--green); animation: pulse-ring 2.2s infinite;
}
.status-badge.err {
  background: var(--red-dim); border: 1px solid rgba(255,77,109,0.3);
  color: var(--red); animation: pulse-ring 2.2s infinite;
}

/* ── BREAKDOWN ─────────────────────────────── */
.breakdown-wrap {
  margin-top: 14px; padding-top: 13px;
  border-top: 1px solid rgba(0,210,255,0.1);
}
.breakdown-head {
  font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
  color: var(--muted); margin-bottom: 9px;
}
.b-row {
  display: flex; align-items: center;
  font-size: 11px; margin-bottom: 7px; gap: 5px;
}
.b-name { min-width: 90px; color: var(--muted); }
.b-track {
  flex: 1; height: 4px; background: rgba(255,255,255,0.06);
  border-radius: 99px; overflow: hidden;
}
.b-fill { height: 100%; border-radius: 99px; background: rgba(0,210,255,0.22); }
.b-fill.active { background: linear-gradient(90deg, var(--teal), var(--cyan)); }
.b-pct { min-width: 34px; text-align: right; color: var(--muted); }

/* ── SECTION HEAD ──────────────────────────── */
.section-head {
  font-family: var(--font-head) !important;
  font-size: 11px; font-weight: 600; letter-spacing: 3px; text-transform: uppercase;
  color: var(--muted); margin: 26px 0 14px;
  display: flex; align-items: center; gap: 10px;
}
.section-head::after { content: ''; flex: 1; height: 1px; background: var(--border); }

.scan-num {
  font-size: 10px; color: var(--muted);
  text-align: right; margin-top: 12px; letter-spacing: 1px;
}

/* ── SIDEBAR LABELS ────────────────────────── */
.sb-logo {
  font-family: var(--font-head) !important;
  font-size: 19px; font-weight: 800; color: var(--cyan); letter-spacing: -0.5px;
}
.sb-sub { font-size: 10px; color: var(--muted); letter-spacing: 2px; text-transform: uppercase; }
.sb-section {
  font-size: 10px; color: var(--muted);
  letter-spacing: 2px; text-transform: uppercase; margin: 16px 0 7px;
}
.sb-class {
  display: flex; align-items: center; gap: 8px;
  padding: 6px 11px; margin-bottom: 5px; border-radius: 8px;
  background: rgba(12,32,54,0.55); border: 1px solid rgba(0,210,255,0.09);
  font-size: 12px;
}
.sb-info {
  background: rgba(0,210,255,0.08); border: 1px solid rgba(0,210,255,0.14);
  border-radius: 9px; padding: 10px 12px; font-size: 11px;
  color: var(--muted); line-height: 1.65; margin-top: 10px;
}

/* ── EMPTY STATE ───────────────────────────── */
.empty-state { text-align: center; padding: 70px 20px; animation: fadeUp 0.6s ease both; }

/* ── FOOTER ────────────────────────────────── */
.app-footer {
  text-align: center; padding: 30px 0 10px;
  font-size: 11px; color: var(--muted); letter-spacing: 0.4px;
}
.app-footer span { color: var(--cyan); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
CLASS_ICONS = ['🔴', '🟠', '🟢', '🟡']


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def get_interpreter():
    interpreter = tf.lite.Interpreter(model_path="model_optimized.tflite")
    interpreter.allocate_tensors()
    return interpreter


def run_inference(image: Image.Image) -> np.ndarray:
    interpreter = get_interpreter()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()
    img = image.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    interpreter.set_tensor(inp[0]['index'], arr)
    interpreter.invoke()
    return interpreter.get_tensor(out[0]['index'])[0]


def pil_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">
    <span class="hero-badge-dot"></span>
    AI · Medical Imaging
  </div>
  <h1 class="hero-title">MRI Scan<br>Diagnostic System</h1>
  <p class="hero-sub">
    Deep-learning MRI classifier &nbsp;·&nbsp; Glioma &nbsp;·&nbsp;
    Meningioma &nbsp;·&nbsp; Pituitary &nbsp;·&nbsp; No Tumor<br>
    Upload one or more axial MRI scans for instant AI-assisted triage.
  </p>
  <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:18px 0 10px">
      <p class="sb-logo">⚕ MRI Scan</p>
      <p class="sb-sub" style="margin:3px 0 18px">Control Panel</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sb-section">Input Method</p>', unsafe_allow_html=True)
    option = st.radio("", ["📂  Upload MRI Files", "📷  Live Camera"],
                      label_visibility="collapsed")

    st.markdown("""
    <div class="sb-info">
      ℹ️ Use <b>T1-weighted axial</b> MRI slices, JPG or PNG,
      min 224×224 px for best accuracy.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sb-section" style="margin-top:18px">Supported Classes</p>',
                unsafe_allow_html=True)
    for icon, name in zip(CLASS_ICONS, CLASS_NAMES):
        st.markdown(f'<div class="sb-class">{icon}&nbsp;{name}</div>',
                    unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────
all_images: list[Image.Image] = []

if option == "📂  Upload MRI Files":
    uploaded = st.file_uploader(
        "Drop MRI scans here or click to browse",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    if uploaded:
        for f in uploaded:
            all_images.append(Image.open(f).convert("RGB"))
else:
    cam = st.camera_input("Capture MRI scan")
    if cam:
        all_images.append(Image.open(cam).convert("RGB"))


# ─────────────────────────────────────────────
# CARD RENDERER  — each section is its own
# st.markdown call to prevent HTML escaping
# ─────────────────────────────────────────────
def render_card(img: Image.Image, probs: np.ndarray, scan_num: int, delay: float):
    idx        = int(np.argmax(probs))
    conf       = float(probs[idx]) * 100
    label      = CLASS_NAMES[idx]
    is_ok      = label == "No Tumor"
    dx_cls     = "clear"  if is_ok else "tumor"
    fill_cls   = "ok"     if is_ok else "danger"
    badge_cls  = "ok"     if is_ok else "err"
    status_ico = "✅"     if is_ok else "⚠️"
    status_msg = ("No malignancy detected"
                  if is_ok else "Suspected neoplasm — refer for review")
    b64 = pil_to_b64(img)

    # 1 · Card shell + image + top metrics
    st.markdown(f"""
<div class="result-card" style="animation-delay:{delay:.2f}s">
  <div class="img-wrapper">
    <img src="data:image/jpeg;base64,{b64}" alt="MRI scan {scan_num}"/>
  </div>
  <p class="dx-label {dx_cls}">{CLASS_ICONS[idx]}&nbsp;{label}</p>
  <p class="dx-conf">CONFIDENCE &nbsp;·&nbsp; {conf:.1f}%</p>
  <div class="conf-track">
    <div class="conf-fill {fill_cls}" style="width:{conf:.1f}%"></div>
  </div>
  <div class="status-badge {badge_cls}">{status_ico}&nbsp;{status_msg}</div>
  <div class="breakdown-wrap">
    <p class="breakdown-head">Class Probabilities</p>
""", unsafe_allow_html=True)

    # 2 · Each probability row as its own markdown block
    #     (avoids Python f-string HTML-escaping inner tags)
    for j, (cname, prob) in enumerate(zip(CLASS_NAMES, probs)):
        pct     = float(prob) * 100
        act_cls = "active" if j == idx else ""
        st.markdown(
            f'<div class="b-row">'
            f'<span class="b-name">{CLASS_ICONS[j]}&nbsp;{cname}</span>'
            f'<div class="b-track"><div class="b-fill {act_cls}" style="width:{pct:.1f}%"></div></div>'
            f'<span class="b-pct">{pct:.1f}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # 3 · Close breakdown + card
    st.markdown(
        f'  </div>\n  <p class="scan-num">SCAN #{scan_num:02d}</p>\n</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
if all_images:
    count = len(all_images)
    st.markdown(
        f'<div class="section-head">Analysis Results &nbsp;·&nbsp; '
        f'{count} scan{"s" if count > 1 else ""}</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2, gap="large")
    for i, img in enumerate(all_images):
        with cols[i % 2]:
            with st.spinner("Analysing scan…"):
                probs = run_inference(img)
            render_card(img, probs, scan_num=i + 1, delay=i * 0.1)

else:
    st.markdown("""
<div class="empty-state">
  <div style="font-size:52px;margin-bottom:14px">🧠</div>
  <p style="font-family:'Syne',sans-serif;font-size:21px;font-weight:700;
            color:#d4eaf7;margin-bottom:7px">No scans loaded</p>
  <p style="font-size:13px;color:#5a8aaa;max-width:330px;
            margin:0 auto;line-height:1.75">
    Upload one or more MRI images using the sidebar to begin
    AI-assisted diagnostic triage.
  </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  MRI Scan &nbsp;·&nbsp; Brain Tumor AI Diagnostic &nbsp;|&nbsp;
  Developed by <span>Khaled Mousa</span> &nbsp;|&nbsp;
  Strategic Architecture &amp; AI Engineering<br>
  <span style="font-size:10px;opacity:0.4">
    ⚠ For research &amp; educational purposes only —
    not a substitute for clinical diagnosis.
  </span>
</div>
""", unsafe_allow_html=True)
