# egitim_kod.py
# Amaç: Derin öğrenme ile Türkçe metin verisini eğitmek ve sesli örnek üzerinden duygu tahmini yapmak

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import tensorflow as tf
import speech_recognition as sr
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from sklearn.model_selection import train_test_split

# --- Veri Yükleme ve Temizleme ---
data = pd.read_csv("_data.csv", delimiter="\t", quoting=3)
data = data[['Review', 'Sentiment']].dropna()
turkish_chars = "a-zA-ZçÇğĞıİöÖşŞüÜ0-9"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(f"[^{turkish_chars}\\s]", "", text)
    return text

data['Review'] = data['Review'].apply(clean_text)

# --- Tokenizer ve Sekanslama ---
max_words = 25000
max_len = 400
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data['Review'])
X = pad_sequences(tokenizer.texts_to_sequences(data['Review']), maxlen=max_len)
Y = pd.get_dummies(data['Sentiment']).values  # 2 sınıflı softmax için one-hot encoding

# --- Eğitim / Test Bölünmesi ---
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# --- LSTM Modeli ---
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    LSTM(64, dropout=0.3, recurrent_dropout=0.3),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# --- Model Eğitimi ---
history = model.fit(X_train, Y_train, epochs=5, batch_size=32,
                    validation_data=(X_test, Y_test), verbose=1)

# --- Test Başarımı ---
loss, acc = model.evaluate(X_test, Y_test, verbose=1)
print("Test Loss:", loss)
print("Test Accuracy:", acc)

# --- Modeli Kaydet ---
model.save("sentiment_analysis_model.h5")

# --- Sesli Örnekle Tahmin Fonksiyonu ---
def perform_sentiment_analysis(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio_data, language="tr-TR")
        print("Sesli Metin:", text)
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=max_len)
        pred = model.predict(padded)[0]
        print("Olumsuz Olasılık:", pred[0])
        print("Olumlu Olasılık:", pred[1])
        return pred
    except sr.UnknownValueError:
        print("Ses anlaşılamadı.")
        return None
    except sr.RequestError as e:
        print("Google API hatası:", e)
        return None

# Örnek ses analizi çağrısı
perform_sentiment_analysis("res.wav")

# --- Eğitim Grafiği ---
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.title('Model Eğitim Performansı')
plt.legend()
plt.show()
