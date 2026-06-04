import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import numpy as np
import os
import io
import plotly.graph_objects as go
import plotly.express as px

# ─── Sayfa Ayarları ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrashNet — Akıllı Atık Sınıflandırıcı",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Sınıf Bilgileri ─────────────────────────────────────────────────────────────
CLASS_INFO = {
    "cardboard": {
        "tr": "Karton",
        "emoji": "📦",
        "color": "#F59E0B",
        "bg": "#FEF3C7",
        "tip": "Islak kartonları kurutun, metal zımbaları çıkarın.",
        "bin": "Mavi (Kağıt/Karton) Kutu",
        "recycle": True,
    },
    "glass": {
        "tr": "Cam",
        "emoji": "🍶",
        "color": "#06B6D4",
        "bg": "#CFFAFE",
        "tip": "Kapaklarını çıkarın, çevrilmeden atın.",
        "bin": "Yeşil (Cam) Kutu",
        "recycle": True,
    },
    "metal": {
        "tr": "Metal",
        "emoji": "🥫",
        "color": "#6366F1",
        "bg": "#EEF2FF",
        "tip": "Temizleyin, ezin (yer kazanır).",
        "bin": "Sarı (Metal) Kutu",
        "recycle": True,
    },
    "paper": {
        "tr": "Kağıt",
        "emoji": "📄",
        "color": "#3B82F6",
        "bg": "#DBEAFE",
        "tip": "Lamine veya yağlı kağıtlar geri dönüşmez.",
        "bin": "Mavi (Kağıt/Karton) Kutu",
        "recycle": True,
    },
    "plastic": {
        "tr": "Plastik",
        "emoji": "🧴",
        "color": "#10B981",
        "bg": "#D1FAE5",
        "tip": "Alt üçgen içindeki numaraya bakın (1-7).",
        "bin": "Sarı (Plastik) Kutu",
        "recycle": True,
    },
    "trash": {
        "tr": "Çöp / Diğer",
        "emoji": "🗑️",
        "color": "#EF4444",
        "bg": "#FEE2E2",
        "tip": "Geri dönüşüm kutusuna atmayın, genel çöp.",
        "bin": "Siyah / Genel Çöp Kutu",
        "recycle": False,
    },
}

CLASS_NAMES = list(CLASS_INFO.keys())   # ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ─── Model Yükleme (cache) ────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.pth")

@st.cache_resource(show_spinner=False)
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 6)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

# ─── Görüntü Dönüşüm ─────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image: Image.Image, model):
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().numpy()
    top_idx  = int(np.argmax(probs))
    top_conf = float(probs[top_idx]) * 100
    return top_idx, top_conf, probs

# ─── CSS Stilleri ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Ana arka plan */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* Başlık alanı */
.hero-title {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.hero-title h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4ff, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
}
.hero-title p {
    color: #8b949e;
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

/* Upload alanı */
.upload-zone {
    border: 2px dashed #30363d;
    border-radius: 16px;
    padding: 3rem;
    text-align: center;
    background: rgba(255,255,255,0.02);
    transition: all 0.3s ease;
    margin: 1rem 0;
}
.upload-zone:hover {
    border-color: #00d4ff;
    background: rgba(0,212,255,0.05);
}

/* Sonuç kartı */
.result-card {
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
}

/* Confidence bar */
.conf-bar-outer {
    background: rgba(255,255,255,0.08);
    border-radius: 100px;
    height: 10px;
    width: 100%;
    margin: 4px 0 12px 0;
    overflow: hidden;
}
.conf-bar-inner {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #00d4ff, #a78bfa);
    transition: width 0.8s ease;
}

/* Metrik kartları */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-card .value { font-size: 1.8rem; font-weight: 700; color: #00d4ff; }
.metric-card .label { font-size: 0.8rem; color: #8b949e; margin-top: 4px; }

/* Info badge */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-green { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.badge-red   { background: rgba(239, 68, 68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important;
    width: 100%;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,212,255,0.3) !important;
}

div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 2px dashed #30363d !important;
    border-radius: 16px !important;
    padding: 1rem !important;
}

/* Divider */
hr { border-color: #30363d !important; }

/* Text colors */
h1,h2,h3,h4 { color: #e6edf3 !important; }
p, span, li  { color: #c9d1d9 !important; }

.stSuccess { background: rgba(52,211,153,0.1) !important; border: 1px solid rgba(52,211,153,0.3) !important; }
.stInfo    { background: rgba(0,212,255,0.1) !important; border: 1px solid rgba(0,212,255,0.3) !important; }
.stWarning { background: rgba(245,158,11,0.1) !important; border: 1px solid rgba(245,158,11,0.3) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ♻️ TrashNet")
    st.markdown("---")
    st.markdown("### 📊 Model Bilgisi")
    st.markdown("""
    | Özellik | Değer |
    |---------|-------|
    | Mimari | ResNet18 |
    | Sınıf | 6 |
    | Val Acc | **%93** |
    | Test Acc | **%89** |
    | Dataset | 2.527 img |
    """)
    st.markdown("---")
    st.markdown("### 🗂️ Sınıflar")
    for key, info in CLASS_INFO.items():
        recycled = "♻️" if info["recycle"] else "🗑️"
        st.markdown(f"{info['emoji']} **{info['tr']}** {recycled}")
    st.markdown("---")
    st.markdown("### ⚙️ Ayarlar")
    show_all_probs = st.toggle("Tüm olasılıkları göster", value=True)
    show_tips = st.toggle("Geri dönüşüm ipuçları", value=True)
    st.markdown("---")
    st.caption("👤 Abdulkadir Doğan · YMG-4 · 2024")

# ─── Ana İçerik ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-title">
    <h1>🗑️ TrashNet</h1>
    <p>Transfer Learning ile Akıllı Atık Sınıflandırma · ResNet18 · %89 Test Accuracy</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Model yükle
with st.spinner("🔄 Model yükleniyor..."):
    try:
        model = load_model()
        model_loaded = True
    except Exception as e:
        st.error(f"❌ Model yüklenemedi: {e}\n\n`models/best_model.pth` dosyasının varlığını kontrol edin.")
        model_loaded = False

col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("### 📤 Görüntü Yükle")
    uploaded = st.file_uploader(
        "Bir atık fotoğrafı seç (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed"
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption=f"📁 {uploaded.name}", use_container_width=True)

        # Görüntü meta bilgileri
        st.markdown("#### 🖼️ Görüntü Bilgisi")
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            st.markdown(f'<div class="metric-card"><div class="value">{image.width}</div><div class="label">Genişlik (px)</div></div>', unsafe_allow_html=True)
        with ic2:
            st.markdown(f'<div class="metric-card"><div class="value">{image.height}</div><div class="label">Yükseklik (px)</div></div>', unsafe_allow_html=True)
        with ic3:
            size_kb = len(uploaded.getvalue()) // 1024
            st.markdown(f'<div class="metric-card"><div class="value">{size_kb}</div><div class="label">Boyut (KB)</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 Sınıflandır", use_container_width=True)
    else:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size: 3rem">📸</div>
            <p style="color:#8b949e; margin-top:1rem">Görüntü yüklemek için tıklayın veya sürükleyip bırakın</p>
            <p style="color:#484f58; font-size:0.85rem">JPG · PNG · WEBP · BMP</p>
        </div>
        """, unsafe_allow_html=True)
        predict_btn = False

# ─── Tahmin & Sonuç ──────────────────────────────────────────────────────────────
with col_result:
    st.markdown("### 🎯 Tahmin Sonucu")

    if uploaded and predict_btn and model_loaded:
        with st.spinner("🧠 Model analiz ediyor..."):
            top_idx, top_conf, all_probs = predict(image, model)

        cls_key  = CLASS_NAMES[top_idx]
        info     = CLASS_INFO[cls_key]
        recycled = info["recycle"]

        # Ana sonuç kartı
        badge_html = '<span class="badge badge-green">♻️ Geri Dönüşebilir</span>' if recycled else '<span class="badge badge-red">🚫 Geri Dönüşemez</span>'
        st.markdown(f"""
        <div class="result-card" style="background: linear-gradient(135deg, {info['bg']}10, {info['color']}10); border-color: {info['color']}40;">
            <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem;">
                <span style="font-size:4rem">{info['emoji']}</span>
                <div>
                    <div style="font-size:0.8rem; color:#8b949e; text-transform:uppercase; letter-spacing:2px">TAHMİN</div>
                    <div style="font-size:2rem; font-weight:800; color:{info['color']}">{info['tr']}</div>
                    <div style="margin-top:4px">{badge_html}</div>
                </div>
            </div>
            <div style="color:#8b949e; font-size:0.9rem; margin-bottom:0.5rem">Güven Skoru</div>
            <div style="font-size:2.5rem; font-weight:800; color:{info['color']}">{top_conf:.1f}%</div>
            <div class="conf-bar-outer">
                <div class="conf-bar-inner" style="width:{top_conf:.1f}%; background: linear-gradient(90deg, {info['color']}, {info['color']}aa);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Geri dönüşüm kutusu bilgisi
        if show_tips:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius:12px; padding:1rem; margin-bottom:1rem;">
                <div style="font-weight:600; color:#e6edf3; margin-bottom:0.5rem">🗺️ Nereye Atılmalı?</div>
                <div style="color:#8b949e; font-size:0.9rem">📍 {info['bin']}</div>
                <div style="margin-top:0.5rem; padding: 0.5rem; background:rgba(255,255,255,0.04); border-radius:8px;">
                    <span style="color:#f59e0b; font-size:0.85rem">💡 {info['tip']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Tüm olasılıklar (Plotly bar chart)
        if show_all_probs:
            st.markdown("#### 📊 Tüm Sınıf Olasılıkları")

            labels  = [CLASS_INFO[cn]["tr"] + " " + CLASS_INFO[cn]["emoji"] for cn in CLASS_NAMES]
            values  = [float(p) * 100 for p in all_probs]
            colors  = [CLASS_INFO[cn]["color"] for cn in CLASS_NAMES]
            # Seçilen sınıfı vurgula
            opacities = [1.0 if i == top_idx else 0.4 for i in range(len(CLASS_NAMES))]
            bar_colors = [f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},{opacities[i]})"
                          for i, c in enumerate(colors)]

            fig = go.Figure(go.Bar(
                x=values,
                y=labels,
                orientation='h',
                marker_color=bar_colors,
                text=[f"{v:.1f}%" for v in values],
                textposition='outside',
                textfont=dict(color='white', size=12),
                hovertemplate='<b>%{y}</b><br>Olasılık: %{x:.2f}%<extra></extra>',
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    range=[0, 115],
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.05)',
                    tickfont=dict(color='#8b949e'),
                    showticklabels=False,
                ),
                yaxis=dict(
                    tickfont=dict(color='#e6edf3', size=13),
                    autorange='reversed',
                ),
                margin=dict(l=0, r=60, t=10, b=10),
                height=280,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    elif uploaded and not predict_btn:
        st.markdown("""
        <div style="text-align:center; padding:4rem 2rem; color:#484f58;">
            <div style="font-size:4rem">🔍</div>
            <p style="margin-top:1rem; font-size:1.1rem; color:#8b949e">
                Görüntü yüklendi!<br>
                <strong style="color:#00d4ff">Sınıflandır</strong> butonuna basın.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:4rem 2rem; color:#484f58;">
            <div style="font-size:4rem">🤖</div>
            <p style="margin-top:1rem; font-size:1.1rem; color:#8b949e">
                Soldan bir görüntü yükleyin,<br>yapay zeka analiz etsin.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ─── Alt Bilgi ───────────────────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
stats = [
    ("🏗️ ResNet18", "Transfer Learning"),
    ("📦 2.527", "Eğitim Görüntüsü"),
    ("🎯 %93", "Validation Accuracy"),
    ("✅ %89", "Test Accuracy"),
]
for col, (val, lbl) in zip([c1, c2, c3, c4], stats):
    with col:
        st.markdown(f'<div class="metric-card"><div class="value" style="font-size:1.3rem">{val}</div><div class="label">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("🎓 İnönü Üniversitesi · YMG-4 Dersi · Abdulkadir Doğan · TrashNet — Stanford TrashNet Dataset")
