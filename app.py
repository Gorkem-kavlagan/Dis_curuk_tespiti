"""
Diş Çürüğü Tespit Sistemi - Streamlit Arayüzü
YOLOv8 ile diş röntgenlerinde çürük tespiti

Görkem Kavlağan - 2212503019
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# Sayfa ayarları
st.set_page_config(
    page_title="Diş Çürüğü Tespit Sistemi",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS ile modern stil
st.markdown("""
<style>
    /* Ana başlık */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(120deg, #1E88E5, #7C4DFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px;
        font-weight: bold;
    }
    
    /* Alt başlık */
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    
    /* Sonuç kutuları */
    .success-box {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.3rem;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    
    .danger-box {
        background: linear-gradient(135deg, #f44336 0%, #c62828 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.3rem;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
    }
    
    /* İstatistik kartları */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Buton stili */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 40px;
        border-radius: 30px;
        font-size: 18px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Upload alanı */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 20px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #888;
        padding: 30px;
        margin-top: 50px;
        border-top: 1px solid #eee;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)

# Model yükleme fonksiyonu
@st.cache_resource
def load_model():
    """Eğitilmiş modeli yükle"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "dental_caries_model", "weights", "best.pt")
    
    if os.path.exists(model_path):
        return YOLO(model_path), True
    else:
        return None, False

# Modeli yükle
model, model_loaded = load_model()

# Sınıf isimleri (Türkçe)
CLASS_NAMES = {
    0: "Restorasyon",     # Tedavi edilmiş/dolgulu bölge
    1: "Çürük"            # Aktif çürük bölgesi
}

# Sınıf renkleri (RGB formatında)
CLASS_COLORS = {
    0: (255, 165, 0),     # Turuncu - Restorasyon
    1: (255, 0, 0)        # Kırmızı - Çürük
}

# Sidebar
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="font-size: 4rem; margin: 0;">🦷</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.title("⚙️ Ayarlar")
    
    st.markdown("---")
    
    # Güven eşiği
    confidence = st.slider(
        "🎯 Güven Eşiği",
        min_value=0.0,
        max_value=0.5,
        value=0.05,
        step=0.005,
        help="Düşük değer = daha fazla tespit, Yüksek değer = daha kesin tespit"
    )
    
    st.markdown("---")
    
    # Bilgi kutusu
    st.info("""
    **📖 Nasıl Kullanılır?**
    
    1. Diş röntgeni yükleyin
    2. "Analiz Et" butonuna tıklayın
    3. Sonuçları inceleyin
    
    **💡 İpucu:** Güven eşiğini ayarlayarak tespit hassasiyetini değiştirebilirsiniz.
    """)
    
    st.markdown("---")
    
    # Model durumu
    if model_loaded:
        st.success("✅ Model yüklendi")
    else:
        st.error("❌ Model bulunamadı")
    
    st.markdown("---")
    
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
        <p><strong>Görkem Kavlağan</strong></p>
        <p>2212503019</p>
    </div>
    """, unsafe_allow_html=True)

# Ana Sayfa
st.markdown('<h1 class="main-header">🦷 Diş Çürüğü Tespit Sistemi</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Yapay Zeka Destekli Diş Röntgeni Analizi</p>', unsafe_allow_html=True)

# Model kontrolü
if not model_loaded:
    st.error("""
    ⚠️ **Model bulunamadı!**
    
    Lütfen önce modeli eğitin:
    ```bash
    python train_model.py
    ```
    
    Eğitim tamamlandıktan sonra bu sayfayı yenileyin.
    """)
    st.stop()

st.markdown("---")

# İki sütunlu düzen
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Görüntü Yükle")
    
    uploaded_file = st.file_uploader(
        "Diş röntgeni seçin",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Desteklenen formatlar: JPG, JPEG, PNG, BMP"
    )
    
    if uploaded_file is not None:
        # Görüntüyü yükle ve göster
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Yüklenen Röntgen", use_container_width=True)
        
        # Görüntü bilgileri
        st.markdown(f"""
        <div class="stat-card">
            <strong>Görüntü Bilgileri</strong><br>
            📐 Boyut: {image.size[0]} x {image.size[1]} piksel<br>
            📁 Format: {image.format if image.format else 'N/A'}
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### 🔍 Analiz Sonucu")
    
    if uploaded_file is not None:
        # Analiz butonu
        analyze_button = st.button("🚀 Analiz Et", use_container_width=True)
        
        if analyze_button:
            with st.spinner("🔄 Görüntü analiz ediliyor..."):
                # YOLO ile tahmin yap
                results = model.predict(
                    source=image,
                    conf=confidence,
                    save=False,
                    verbose=False
                )
                
                # Sonuç görüntüsünü özel renklerle oluştur
                import cv2
                result_image = np.array(image)
                if len(result_image.shape) == 2:  # Grayscale ise RGB'ye çevir
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2RGB)
                elif result_image.shape[2] == 4:  # RGBA ise RGB'ye çevir
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGBA2RGB)
                
                detections = results[0].boxes
                num_detections = len(detections)
                
                # Sınıf sayıları
                num_curuk = 0
                num_restorasyon = 0
                
                for box in detections:
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Sınıfa göre renk seç (RGB formatı)
                    if cls_id == 0:
                        color = (255, 165, 0)  # Turuncu - Restorasyon
                        num_restorasyon += 1
                    else:
                        color = (255, 0, 0)  # Kırmızı - Çürük
                        num_curuk += 1
                    
                    # Kutu çiz
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)
                
                st.image(result_image, caption="🎯 Tespit Sonucu", use_container_width=True)
                
                # Renk açıklaması
                st.markdown("""
                <div style="display: flex; gap: 20px; justify-content: center; margin: 10px 0;">
                    <span style="color: #FF0000; font-weight: bold;">🔴 Kırmızı = Çürük</span>
                    <span style="color: #FFA500; font-weight: bold;">🟠 Turuncu = Restorasyon</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                if num_detections > 0:
                    # Tespit özeti
                    st.markdown(f"""
                    <div class="danger-box">
                        ⚠️ <strong>Toplam {num_detections} tespit yapıldı!</strong><br>
                        🔴 Çürük: {num_curuk} adet | 🟠 Restorasyon: {num_restorasyon} adet
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detaylı sonuçlar
                    st.markdown("#### 📋 Detaylı Analiz Raporu")
                    
                    for i, box in enumerate(detections):
                        conf_score = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        
                        # Koordinatlar
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Sınıfa göre emoji ve isim
                        if cls_id == 0:
                            emoji = "🟠"
                            sinif_adi = "Restorasyon"
                        else:
                            emoji = "🔴"
                            sinif_adi = "Çürük"
                        
                        with st.expander(f"{emoji} {sinif_adi} #{i+1} - Güven: %{conf_score*100:.1f}"):
                            st.write(f"**Sınıf:** {sinif_adi}")
                            st.write(f"**Güven Skoru:** %{conf_score*100:.1f}")
                            st.write(f"**Konum:** ({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)})")
                            
                            # Risk seviyesi
                            if conf_score >= 0.7:
                                st.error("🔴 Yüksek güvenle tespit edildi")
                            elif conf_score >= 0.4:
                                st.warning("🟡 Orta güvenle tespit edildi")
                            else:
                                st.info("🟢 Düşük güvenle tespit edildi")
                    
                    st.warning("""
                    **⚠️ Önemli Not:** Bu sistem yalnızca yardımcı bir araçtır. 
                    Kesin teşhis için mutlaka bir diş hekimine danışın.
                    """)
                else:
                    # Çürük tespit edilmedi
                    st.markdown("""
                    <div class="success-box">
                        ✅ <strong>Çürük tespit edilmedi!</strong><br>
                        Görüntüde belirgin bir çürük bölgesi bulunamadı.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.info("""
                    **💡 Not:** Bu sonuç, mevcut güven eşiği ile yapılan analize göre verilmiştir.
                    Farklı sonuçlar için güven eşiğini ayarlayabilirsiniz.
                    """)
    else:
        # Görüntü yüklenmemiş
        st.markdown("""
        <div style="
            text-align: center;
            padding: 50px;
            background: #f8f9fa;
            border-radius: 15px;
            border: 2px dashed #dee2e6;
        ">
            <h3 style="color: #888;">📷</h3>
            <p style="color: #888;">Analiz için sol taraftan bir diş röntgeni yükleyin</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🦷 <strong>Diş Çürüğü Tespit Sistemi</strong></p>
    <p>YOLOv8 + Streamlit ile geliştirilmiştir</p>
    <p style="font-size: 0.8rem;">Görkem Kavlağan - 2212503019 | Görüntü İşleme Dersi</p>
</div>
""", unsafe_allow_html=True)
