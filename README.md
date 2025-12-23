Diş Çürüğü Tespit Sistemi
Bu proje, YOLOv8 derin öğrenme modeli kullanarak diş röntgenlerinde çürük ve restorasyon tespiti yapan bir yapay zeka uygulamasıdır. Görüntü İşleme dersi kapsamında geliştirilmiştir.

Proje Hakkında
Sistem, diş röntgeni görüntülerini analiz ederek çürük bölgelerini ve restorasyonları (dolgular) otomatik olarak tespit eder. Kullanıcı dostu bir web arayüzü ile röntgen görüntüleri yüklenebilir ve anında analiz sonuçları alınabilir.

Kullanılan Teknolojiler
Python 3.x
YOLOv8 (Ultralytics)
Streamlit (Web arayüzü)
OpenCV (Görüntü işleme)
PIL (Görüntü yükleme)
NumPy
Veri Seti
Veri seti Roboflow platformundan alınmıştır. İki sınıf içermektedir: Çürük ve Restorasyon. Toplamda eğitim için yaklaşık 155, doğrulama için 45 ve test için 19 görüntü kullanılmıştır.

Model Eğitimi
Model, YOLOv8 nano mimarisi kullanılarak CPU üzerinde eğitilmiştir. Eğitim parametreleri: 50 epoch, 640x640 görüntü boyutu, 8 batch size. Early stopping ile 10 epoch boyunca iyileşme olmazsa eğitim durur.

Kullanım
Model eğitimi için: python train_model.py

Web uygulamasını başlatmak için: streamlit run app.py

Özellikler
Diş röntgeni yükleme ve analiz
Çürük tespiti (kırmızı kutu ile gösterilir)
Restorasyon tespiti (turuncu kutu ile gösterilir)
Güven eşiği ayarlama özelliği
Detaylı analiz raporu
Modern ve kullanıcı dostu arayüz
Geliştirici
Görkem Kavlağan - 2212503019
