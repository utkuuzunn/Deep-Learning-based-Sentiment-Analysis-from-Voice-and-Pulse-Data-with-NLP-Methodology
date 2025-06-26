# model_deneme_kod.py
# Amaç: Derin öğrenme tabanlı duygu analizi için ANN, Embedding, CNN ve LSTM modellerini karşılaştırmak

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
nltk.download('stopwords')

# --- Veri Yükleme ve Temizleme ---
df = pd.read_csv("_data.csv", encoding="utf-16")
df['Durum'] = df['Durum'].map({'Olumsuz': 0, 'Olumlu': 1})

# Türkçe stopword temizliği
stop_words = set(nltk.corpus.stopwords.words('turkish'))
def clean_text(text):
    text = re.sub(r'[!.,\n,:“”\"?@#\d]', ' ', str(text).lower())
    return ' '.join([word for word in text.split() if word not in stop_words])

df['clean_text'] = df['Görüş'].apply(clean_text)
df.dropna(subset=['clean_text', 'Durum'], inplace=True)

# --- Veri Bölme ---
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['Durum'], test_size=0.2, random_state=42)

# --- Tokenizer ve Pad işlemleri ---
num_words = 10000
maxlen = 200
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(X_train)
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=maxlen)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=maxlen)

early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# --- Model Değerlendirme Yardımcı Fonksiyonu ---
def evaluate_model(name, model, X_test, y_test, results):
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    acc = model.evaluate(X_test, y_test)[1]
    print(f"\n{name} Model Sonuçları")
    print(classification_report(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
    plt.title(name)
    plt.show()
    results.append({'Model': name, 'Accuracy': acc})

# --- Modeller ve Eğitim ---
models = []

# 1. ANN
ann = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(maxlen,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(X_train_seq, y_train, validation_data=(X_test_seq, y_test),
        epochs=20, batch_size=32, callbacks=[early_stopping], verbose=0)
evaluate_model("ANN", ann, X_test_seq, y_test, models)

# 2. Embedding + Global Pooling
embedding = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(maxlen,)),
    tf.keras.layers.Embedding(num_words, 16),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
embedding.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
embedding.fit(X_train_seq, y_train, validation_data=(X_test_seq, y_test),
              epochs=20, batch_size=32, callbacks=[early_stopping], verbose=0)
evaluate_model("Embedding", embedding, X_test_seq, y_test, models)

# 3. CNN
cnn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(maxlen,)),
    tf.keras.layers.Embedding(num_words, 16),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(X_train_seq, y_train, validation_data=(X_test_seq, y_test),
        epochs=20, batch_size=32, callbacks=[early_stopping], verbose=0)
evaluate_model("CNN", cnn, X_test_seq, y_test, models)

# 4. LSTM
lstm = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(maxlen,)),
    tf.keras.layers.Embedding(num_words, 16),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm.fit(X_train_seq, y_train, validation_data=(X_test_seq, y_test),
         epochs=20, batch_size=32, callbacks=[early_stopping], verbose=0)
evaluate_model("LSTM", lstm, X_test_seq, y_test, models)

# --- Sonuç Tablosu ---
result_df = pd.DataFrame(models)
print("\nModel Karşılaştırma Tablosu:")
print(result_df.sort_values("Accuracy", ascending=False))
