# Production Anomaly AI

Üretim verilerindeki anormal davranışları tespit eden ve seçilen kaydı yerel Ollama ile açıklayan bir Python uygulamasıdır.

Bu proje, arıza nedeni teşhisi ya da öngörücü bakım yerine; operasyonel izleme için “normalden sapma” analizi sunar. Geçmiş veriden öğrenen bir model, yeni kaydın anomali olup olmadığını işaretler ve ilgili gözlemleri doğal dil ile açıklar.

## Özellikler

- CSV ile üretim kayıtları yükleme
- Zaman sırasına göre eğitim/test ayrımı
- Isolation Forest tabanlı anomali tespiti
- Makine filtresi, skor grafiği ve işaretlenen kayıt listesi
- İsteğe bağlı makineye özel model; eğitiminde 30'dan az kayıt bulunan makineler ortak modeli kullanır
- Filtrelenen test dönemi için makine bazında işaretlenme oranı ve skor özeti
- Etiket varsa precision, recall, F1 ve confusion matrix hesaplama
- Yerel Ollama ile açıklama üretme
- Ollama kapalı olsa bile temel analiz çalıştırma

## Nasıl Çalışır?

1. CSV verisi doğrulanır.
2. Her kayıt için üretim, hata oranı, duruş oranı ve çalışan dakika başına üretim özellikleri çıkarılır.
3. Geçmiş zaman dilimleri eğitim, sonraki dönemler test için kullanılır.
4. Isolation Forest yalnızca eğitim verisinden öğrenir. Varsayılan ortak modele ek olarak her makine için ayrı model seçilebilir; eğitiminde 30'dan az kayıt bulunan makineler ortak modele geri döner.
5. Test kayıtları anomali olarak işaretlenir.
6. Sonuç ekranı filtrelere uyan kayıtları makine bazında özetler. İşaretlenme oranı arıza olasılığı değildir.
7. Seçilen kayıt, eğitim referanslarıyla birlikte Ollama’ya gönderilir ve Türkçe açıklama üretilir.

## Teknoloji Stack

- Python 3.10+
- Streamlit
- pandas
- scikit-learn
- Plotly
- Ollama (yerel model açıklama için)

## Kurulum

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## Uygulamayı Çalıştırma

```bash
./.venv/Scripts/python.exe -m streamlit run app.py
```

Açılan arayüzde:

- Sentetik demo verisini kullanabilir veya
- kendi CSV dosyanızı yükleyebilirsiniz.

## CSV Veri Formatı

Aşağıdaki kolonlar gereklidir:

```csv
timestamp,machine,production_count,scrap_count,downtime_minutes,shift_minutes
```

Kolon açıklamaları:

- `timestamp`: ISO 8601 tarih/zaman
- `machine`: Makine kimliği
- `production_count`: toplam üretim adedi
- `scrap_count`: hatalı ürün adedi
- `downtime_minutes`: duruş süresi
- `shift_minutes`: toplam vardiya süresi
- `is_anomaly`: isteğe bağlı gerçek etiket (0 veya 1); etiketsiz kullanımda bu kolonu eklemeyin.

İsteğe bağlı etiket yoksa `is_anomaly` sütununu kaldırabilirsiniz. Aynı makine için aynı zamana ait tekrarlı kayıtlar kabul edilmez.

## Ollama Kullanımı

Ollama kurulu ve çalışıyor olmalı. Aşağıdaki komutla mevcut modelleri kontrol edin:

```bash
ollama list
```

Ardından uygulamadaki model alanına kullandığınız model adını yazın. Seçilen kayıt için “Ollama ile açıkla” butonuna basıldığında, yalnızca ilgili kayıt ve geçmiş istatistikler yerel olarak modelinize iletilir.

## Demo Verisi Üretme

```bash
./.venv/Scripts/python.exe demo.py
```

## Test Çalıştırma

```bash
./.venv/Scripts/python.exe -m unittest discover -s tests -v
```

## Sınırlamalar

- Tüm makineler için tek ortak model kullanılır.
- Gerçek üretim bağlamı (ürün tipi, bakım, vardiya türü vb.) henüz kullanılmaz.
- Model çıktısı, kesin arıza nedeni veya gelecekteki arıza tahmini değildir.
- Test verisinde eşik seçimi, veri setine uyum sağlayabilir; bu nedenle gerçek operasyonel kullanım için ek doğrulama önerilir.

## İletişim

Sorularınız, geri bildirimleriniz veya iş birliği fırsatları için dilediğiniz zaman iletişime geçebilirsiniz.

🌐 **Website**  
https://www.yusuftan.com.tr

💼 **LinkedIn**  
https://www.linkedin.com/in/yusuftann/

📧 **Email**  
yusuftan41@hotmail.com