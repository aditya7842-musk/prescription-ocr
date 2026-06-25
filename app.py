"""
Indian Doctor Prescription OCR — Streamlit Frontend
====================================================
Upload a prescription photo → AI reads it → Shows medicines + danger warnings
"""
import os
import tempfile
import streamlit as st
from PIL import Image


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Prescription Safety Checker",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');

    :root {
        --bg:         #11151f;
        --surface:    #1a2030;
        --surface-2:  #212a40;
        --border:     #2d3548;
        --text:       #e8e6e1;
        --text-muted: #8a92a6;
        --accent:     #e8a33d;
        --teal:       #5ec9bd;
        --coral:      #e2725b;
        --green:      #5ec97a;
    }

    .stApp { background: var(--bg); }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text);
        letter-spacing: -0.01em;
    }

    .stButton > button {
        background: var(--accent);
        color: #11151f;
        border: none;
        border-radius: 6px;
        padding: 10px 24px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.15s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: #f3b75a;
        box-shadow: 0 2px 12px rgba(232,163,61,0.3);
    }

    .medicine-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 3px solid var(--teal);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .medicine-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--teal);
        margin: 0 0 4px 0;
    }
    .medicine-detail {
        font-size: 0.88rem;
        color: var(--text-muted);
        margin: 2px 0;
    }

    .warning-high {
        background: rgba(226,114,91,0.08);
        border: 1px solid rgba(226,114,91,0.4);
        border-left: 3px solid var(--coral);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .warning-medium {
        background: rgba(232,163,61,0.08);
        border: 1px solid rgba(232,163,61,0.3);
        border-left: 3px solid var(--accent);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .warning-low {
        background: rgba(94,201,189,0.06);
        border: 1px solid rgba(94,201,189,0.25);
        border-left: 3px solid var(--teal);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }

    .result-box {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 16px 18px;
        margin: 8px 0;
    }
    .ocr-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: var(--teal);
        line-height: 1.6;
        white-space: pre-wrap;
    }

    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 14px 16px;
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent);
    }

    .hero-banner {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: var(--text);
        margin: 0 0 6px 0;
    }
    .hero-sub {
        color: var(--text-muted);
        font-size: 1rem;
        margin: 0;
    }

    .section-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 10px;
    }

    [data-testid="stFileUploader"] {
        background: var(--surface);
        border: 1px dashed var(--border);
        border-radius: 10px;
        padding: 8px;
    }

    code {
        background: var(--surface-2) !important;
        color: var(--teal) !important;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace !important;
        padding: 1px 5px;
    }

    hr { border-color: var(--border); }
    [data-testid="stHeaderActionElements"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-banner">
    <div class="hero-title">💊 Prescription Safety Checker</div>
    <p class="hero-sub">
        Upload a doctor's prescription → AI reads medicine names →
        Warns about dangerous look-alike drugs used in India
    </p>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ How it works", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("**📸 Step 1**\nUpload a photo of any handwritten or printed prescription")
    col2.markdown("**🔍 Step 2**\nEasyOCR (deep learning) reads medicine names from the image")
    col3.markdown("**💊 Step 3**\nFuzzy matching finds correct medicines even with OCR errors")
    col4.markdown("**⚠️ Step 4**\nDanger checker warns about look-alike drug name confusions")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

left_col, right_col = st.columns([1, 1.6], gap="large")


# ─── LEFT COLUMN — Upload ─────────────────────────────────────────────────────

with left_col:

    st.markdown('<div class="section-label">Upload Prescription</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose image",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Supports JPG, PNG, BMP, TIFF",
        label_visibility="collapsed",
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Prescription", use_column_width=True)
        st.markdown(f"""
        <div class="result-box">
            <span style="color:var(--text-muted); font-size:0.85rem;">
            📁 {uploaded_file.name} &nbsp;|&nbsp;
            📐 {image.size[0]}×{image.size[1]}px &nbsp;|&nbsp;
            💾 {uploaded_file.size // 1024} KB
            </span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.82rem; color:var(--text-muted); margin-top:8px;">
        💡 <strong>Tips for better accuracy:</strong><br>
        • Bright, even lighting — no shadows<br>
        • Prescription flat, no folds<br>
        • Full prescription in frame<br>
        • Avoid blurry or rotated images
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── DEMO MODE ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Or Try Demo Mode</div>', unsafe_allow_html=True)
    demo_mode = st.toggle("Type text manually (skip OCR)", value=False)
    demo_text = ""
    if demo_mode:
        demo_text = st.text_area(
            "Paste or type prescription text",
            value="Metformin 500mg BD, Amlodipine 5mg OD, Augmentin 625mg TDS 5 days",
            height=100,
            label_visibility="collapsed",
        )

    st.markdown("---")

    analyze_clicked = st.button(
        "🔍 Analyze Prescription",
        disabled=(uploaded_file is None and not demo_mode),
        use_container_width=True,
    )

    if not uploaded_file and not demo_mode:
        st.markdown(
            '<p style="color:var(--text-muted); font-size:0.85rem; text-align:center;">'
            'Upload an image or enable demo mode</p>',
            unsafe_allow_html=True,
        )


# ─── RIGHT COLUMN — Results ───────────────────────────────────────────────────

with right_col:

    if not uploaded_file and not demo_mode:
        st.markdown("""
        <div style="background:var(--surface); border:1px dashed var(--border);
                    border-radius:12px; padding:48px 32px; text-align:center; margin-top:32px;">
            <div style="font-size:3rem; margin-bottom:12px;">📋</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.2rem;
                        color:var(--text); font-weight:600; margin-bottom:8px;">
                Results will appear here
            </div>
            <div style="color:var(--text-muted); font-size:0.9rem;">
                Upload a prescription or enable demo mode to get started
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif analyze_clicked:

        tmp_path = None

        try:
            # ── STEP 1: Get text (OCR or demo) ────────────────────────────────
            if demo_mode:
                ocr_result = {
                    "raw_text":   demo_text,
                    "lines":      demo_text.split(","),
                    "confidence": 1.0,
                }
            else:
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                with st.spinner("📸 Reading prescription... (CRAFT detector + ResNet18 classifier)"):
                    from core.ocr_engine import extract_text_from_image
                    ocr_result = extract_text_from_image(tmp_path)

                # Only hard-stop if ALL stages failed and we have no text at all
                if not ocr_result.get("raw_text", "").strip():
                    st.error("❌ Could not read text from this image.")
                    st.markdown("""
                    **Try these fixes:**
                    - Take photo in brighter lighting
                    - Keep prescription flat with no shadows
                    - Or use **Demo Mode** on the left to test without an image
                    """)
                    st.stop()

                # Soft warning if only the EasyOCR fallback was used
                if ocr_result.get("engine") == "easyocr":
                    st.info("ℹ️ Medicine classifier not confident — using EasyOCR text for fuzzy matching.")

            # ── STEP 2: Clean text ────────────────────────────────────────────
            with st.spinner("🧹 Cleaning OCR output..."):
                from core.text_cleaner import parse_prescription_text
                parsed = parse_prescription_text(ocr_result["raw_text"])

            # ── STEP 3: Match medicines ───────────────────────────────────────
            with st.spinner("💊 Matching medicine names..."):
                from core.medicine_matcher import find_medicines_in_text
                medicines = find_medicines_in_text(parsed["cleaned_text"])

            # ── STEP 4: Danger checks ─────────────────────────────────────────
            with st.spinner("⚠️ Running safety checks..."):
                from core.danger_checker import run_all_danger_checks
                danger_report = run_all_danger_checks(medicines)

            # ── RESULTS ───────────────────────────────────────────────────────

            st.markdown('<div class="section-label">Analysis Summary</div>', unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("💊 Medicines",    len(medicines))
            m2.metric("⚠️ Warnings",     danger_report["total_warnings"])
            m3.metric("🔴 High Risk",    len(danger_report["high_risk"]))
            ocr_engine = ocr_result.get("engine", "easyocr")
            engine_label = "CRAFT+ResNet18" if ocr_engine == "craft+resnet18" else "EasyOCR" 
            m4.metric("🔍 OCR Engine", engine_label)

            st.markdown("---")

            # Extracted text
            st.markdown('<div class="section-label">Extracted Text</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="result-box">
                <div class="ocr-text">{ocr_result['raw_text']}</div>
            </div>
            """, unsafe_allow_html=True)

            detail_parts = []
            if parsed["dosages"]:
                detail_parts.append(f"**Dosages:** {' · '.join(parsed['dosages'])}")
            if parsed["frequencies"]:
                detail_parts.append(f"**Frequency:** {' · '.join(parsed['frequencies'])}")
            if parsed["durations"]:
                detail_parts.append(f"**Duration:** {' · '.join(parsed['durations'])}")
            if detail_parts:
                st.markdown("  &nbsp;&nbsp;".join(detail_parts))

            st.markdown("---")

            # Medicines found
            st.markdown('<div class="section-label">Medicines Identified</div>', unsafe_allow_html=True)

            if medicines:
                for med in medicines:
                    score = med["match_score"]
                    score_color = "#5ec9bd" if score >= 90 else "#e8a33d" if score >= 75 else "#e2725b"
                    st.markdown(f"""
                    <div class="medicine-card">
                        <span style="float:right; font-family:'JetBrains Mono',monospace;
                                     font-size:0.8rem; color:{score_color};">{score}% match</span>
                        <div class="medicine-name">{med['matched_name']}</div>
                        <div class="medicine-detail">Generic: <strong>{med['generic_name']}</strong></div>
                        <div class="medicine-detail">Category: {med['category']}</div>
                        <div class="medicine-detail" style="font-size:0.78rem;">
                            OCR read: "{med['found_name']}"
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No medicines identified. Try a clearer image or use demo mode.")

            st.markdown("---")

            # Danger warnings
            st.markdown('<div class="section-label">Safety Warnings</div>', unsafe_allow_html=True)

            if danger_report["total_warnings"] == 0:
                st.markdown("""
                <div style="background:rgba(94,201,122,0.08); border:1px solid rgba(94,201,122,0.3);
                            border-radius:10px; padding:14px 16px;">
                    ✅ <strong style="color:#5ec97a;">No dangerous look-alike warnings found</strong>
                    <div style="color:var(--text-muted); font-size:0.85rem; margin-top:4px;">
                        Always verify with your chemist regardless.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for w in danger_report["high_risk"]:
                    st.markdown(f"""
                    <div class="warning-high">
                        <div style="font-weight:600; font-family:'Space Grotesk',sans-serif;
                                    color:var(--coral); margin-bottom:6px;">
                            {w['severity']} — Look-alike Drug Warning
                        </div>
                        <div style="font-size:0.9rem; color:var(--text);">{w['message']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                for w in danger_report["moderate_risk"]:
                    st.markdown(f"""
                    <div class="warning-medium">
                        <div style="font-weight:600; font-family:'Space Grotesk',sans-serif;
                                    color:var(--accent); margin-bottom:6px;">
                            {w['severity']} — Similar Name Alert
                        </div>
                        <div style="font-size:0.9rem; color:var(--text);">{w['message']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                for w in danger_report["caution"]:
                    st.markdown(f"""
                    <div class="warning-low">
                        <div style="font-weight:600; font-family:'Space Grotesk',sans-serif;
                                    color:var(--teal); margin-bottom:6px;">
                            {w['severity']} — Same Category
                        </div>
                        <div style="font-size:0.9rem; color:var(--text);">{w['message']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("""
            <div style="font-size:0.78rem; color:var(--text-muted); text-align:center;
                        background:var(--surface); border-radius:8px; padding:10px 14px;">
                ⚠️ This tool assists in identifying medicines — it does NOT replace medical advice.
                Always verify with your doctor or licensed chemist before consumption.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.info("Make sure all project files are in the same folder as app.py")

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    elif (uploaded_file or demo_mode) and not analyze_clicked:
        st.markdown("""
        <div style="background:var(--surface); border:1px solid var(--border);
                    border-radius:12px; padding:40px 32px; text-align:center; margin-top:32px;">
            <div style="font-size:2.5rem; margin-bottom:12px;">👈</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem;
                        color:var(--text); font-weight:600;">
                Click "Analyze Prescription" to start
            </div>
        </div>
        """, unsafe_allow_html=True)
