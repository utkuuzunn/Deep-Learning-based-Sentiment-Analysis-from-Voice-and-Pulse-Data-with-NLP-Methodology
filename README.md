# Derin Öğrenme Tabanlı Doğal Dil İşleme ile Ses ve Nabız Verilerinden Duygu Durumu Analizi

Bu proje, kullanıcının sesli ifadesini yazıya çevirerek duygu durum analizini yapar ve aynı anda Arduino üzerinden alınan nabız verisiyle duygu durumu arasında ilişki kurar.

## Özellikler
- Ses dosyasından metin çıkarımı
- Derin öğrenme modeli ile metinden duygu analizi 
- Duygu sınıflandırma (olumlu / olumsuz / nötr)
- Nabız değerine göre fizyolojik durum yorumları
- Flask tabanlı web arayüzü
- Arduino ile gerçek zamanlı nabız ölçümü

## Kullanılan Modeller

Proje kapsamında farklı modeller karşılaştırılmıştır:

- Yapay Sinir Ağı (ANN)
- Embedding + Dense
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)

## Kurulum
```
git clone https://github.com/utkuuzunn/duygu-durumu-analizi.git
cd duygu-durumu-analizi
pip install -r requirements.txt
python app.py
```

## Proje Yapısı
- app.py: Ana uygulama
- notebooks/: Model eğitimi ve karşılaştırması
- arduino/: Nabız ölçüm devresi kodu
- templates/: HTML şablonlar
- static/: CSS ve görseller
- model/: Eğitim verisi ve model dosyası

## Ekip
- İsmet Eren Coşkun
- Utku Uzunhüseyin
- Muhammet Emir Orhan
- Melih Durkun
