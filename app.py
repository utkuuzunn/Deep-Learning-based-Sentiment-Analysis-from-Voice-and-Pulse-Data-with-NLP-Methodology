from flask import Flask, request, render_template
import os
import speech_recognition as sr
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import re
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Gerekli klasör varsa oluştur
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Türkçe karakter filtreleme için regex karakter kümesi
turkish_chars = "a-zA-ZçÇğĞıİöÖşŞüÜ0-9"

# Nabız değerine göre yorumlama tablosu
emotion_table = {
    (50, 60): "Dinlenme/Rahatlama",
    (60, 80): "Huzur/Mutluluk",
    (80, 90): "Normal/Nötr",
    (90, 100): "Hafif Stres/Kaygı/Heyecan",
    (100, 110): "Yüksek Stres/Korku/Panik",
    (110, 220): "Aşırı Heyecan/Şiddetli Korku"
}

# Model ve tokenizer yükleniyor
try:
    tokenizer = Tokenizer(num_words=25000)
    data = pd.read_csv("static/uploads/_data.csv", delimiter="\t", quoting=3)
    data['Review'] = data['Review'].astype(str).apply(lambda x: x.lower())
    data['Review'] = data['Review'].apply(lambda x: re.sub(f'[^ {turkish_chars}\\s]', '', x))
    tokenizer.fit_on_texts(data['Review'].values)
    model = tf.keras.models.load_model('static/uploads/sentiment_analysis_model.h5')
except Exception as e:
    print("Model veya veri yüklenemedi:", e)
    model = None

# Nabız aralığını duygu durumuna çevir
def get_emotion_from_heart_rate(heart_rate):
    for range_tuple, emotion in emotion_table.items():
        if range_tuple[0] <= heart_rate < range_tuple[1]:
            return emotion
    return None

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    if 'audio' not in request.files:
        return render_template("error.html", error="Ses dosyası bulunamadı.")

    audio_file = request.files['audio']
    if audio_file.filename == '' or not audio_file.filename.endswith('.wav'):
        return render_template("error.html", error="Geçersiz ses dosyası formatı.")

    try:
        heart_rate = float(request.form.get('heart_rate'))
        if heart_rate < 50 or heart_rate > 220:
            raise ValueError()
    except ValueError:
        return render_template("error.html", error="Hatalı nabız değeri girdiniz.")

    filename = secure_filename(audio_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(file_path)

    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio_data, language="tr-TR")
        sequences = tokenizer.texts_to_sequences([text])
        padded_data = pad_sequences(sequences, maxlen=400)
        prediction = model.predict(padded_data)

        positive_prob = prediction[0][1] * 100
        negative_prob = prediction[0][0] * 100

        emotion = get_emotion_from_heart_rate(heart_rate)
        if emotion is None:
            return render_template("error.html", error="Nabız aralığı geçersiz.")

        if 45 <= positive_prob <= 55:
            sentiment = "Nötr"
            color = "blue"
        elif positive_prob > negative_prob:
            sentiment = "Olumlu"
            color = "green"
        else:
            sentiment = "Olumsuz"
            color = "red"

        result = f"Kişi {emotion} durumunda ve cümlesi {sentiment}."
        return render_template("result.html", result=result, text=text,
                               positive_prob=f"{positive_prob:.2f}",
                               negative_prob=f"{negative_prob:.2f}",
                               color=color)
    except sr.UnknownValueError:
        return render_template("error.html", error="Ses anlaşılamadı.")
    except sr.RequestError as e:
        return render_template("error.html", error=f"Google API hatası: {e}")

if __name__ == '__main__':
    app.run(debug=True)
