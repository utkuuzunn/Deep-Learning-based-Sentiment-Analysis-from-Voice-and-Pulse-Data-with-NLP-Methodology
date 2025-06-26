# Deep Learning-Based Sentiment Analysis from Voice and Pulse Data

This project analyzes the user's emotional state by converting spoken expressions into text and performing sentiment analysis, while simultaneously correlating the results with real-time pulse data collected via Arduino.

## Features
- Speech-to-text conversion from audio files
- Sentiment analysis from text using deep learning models 
- Emotion classification (positive / negative / neutral)
- Interpretation of physiological state based on pulse data
- Web interface built with Flask
- Real-time heart rate monitoring via Arduino

## Models Used

Several models were trained and compared during the project:

- Artificial Neural Network (ANN)
- Embedding + Dense Layers
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)

## Installation
```
git clone https://github.com/utkuuzunn/Deep-Learning-based-Sentiment-Analysis-from-Voice-and-Pulse-Data-with-NLP-Methodology
cd Deep-Learning-based-Sentiment-Analysis-from-Voice-and-Pulse-Data-with-NLP-Methodology
pip install -r requirements.txt
python app.py
```

## Project Structure
- app.py: Main application file
- notebooks/: Model training and comparison notebooks
- arduino/: Code for the heart rate monitoring circuit
- templates/: HTML templates
- static/: CSS and image assets
- model/: Training data and model files

## Project Team Members
- İsmet Eren Coşkun
- Utku Uzunhüseyin
- Muhammet Emir Orhan
- Melih Durkun
