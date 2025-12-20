"""
Diş Çürüğü Tespit Modeli Eğitim Scripti
YOLOv8 ile diş röntgenlerinde çürük tespiti

Görkem Kavlağan - 2212503019
"""

from ultralytics import YOLO
import os

def main():
    # Proje dizini
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_YAML = os.path.join(BASE_DIR, "dataset", "data.yaml")
    
    print("=" * 50)
    print("🦷 Diş Çürüğü Tespit Modeli Eğitimi")
    print("=" * 50)
    
    # data.yaml dosyasını kontrol et
    if not os.path.exists(DATA_YAML):
        print(f"❌ Hata: {DATA_YAML} bulunamadı!")
        print("Lütfen dataset klasörünün doğru konumda olduğundan emin olun.")
        return
    
    print(f"\n📁 Veri seti: {DATA_YAML}")
    print(f"📂 Model kaydedilecek: {BASE_DIR}/dental_caries_model/")
    
    # YOLOv8 nano modeli (CPU için en hızlı ve verimli)
    print("\n📥 YOLOv8n modeli yükleniyor...")
    model = YOLO("yolov8n.pt")
    
    print("\n🚀 Eğitim başlıyor...")
    print("⚠️  CPU ile eğitim yapılacak, bu işlem uzun sürebilir.")
    print("-" * 50)
    
    # Modeli eğit
    results = model.train(
        data=DATA_YAML,
        epochs=50,              # Epoch sayısı
        imgsz=640,              # Görüntü boyutu
        batch=8,                # CPU için düşük batch size
        device="cpu",           # CPU kullanımı (ekran kartı yok)
        project=BASE_DIR,
        name="dental_caries_model",
        patience=10,            # Early stopping - 10 epoch iyileşme olmazsa dur
        save=True,              # Modeli kaydet
        plots=True,             # Grafikleri oluştur
        verbose=True,           # Detaylı çıktı
        exist_ok=True           # Klasör varsa üzerine yaz
    )
    
    print("\n" + "=" * 50)
    print("✅ Eğitim tamamlandı!")
    print("=" * 50)
    
    best_model_path = os.path.join(BASE_DIR, "dental_caries_model", "weights", "best.pt")
    print(f"\n📦 En iyi model: {best_model_path}")
    print("\n💡 Şimdi Streamlit uygulamasını başlatabilirsiniz:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    main()
