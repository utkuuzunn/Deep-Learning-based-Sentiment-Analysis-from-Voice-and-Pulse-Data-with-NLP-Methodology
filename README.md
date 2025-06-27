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

## Notes on Dataset and Pretrained Model

This project was originally trained using a Turkish sentiment dataset, which is **not publicly shared** due to privacy and size restrictions.

However, you can use **any sentiment dataset** in CSV format with two columns:
- `text` (or `review`)
- `label` (e.g., Positive/Negative or 1/0)

To retrain the model with your own data, refer to the notebook:  
`notebooks/model_training.py`

---

If you want to use the **pre-trained model directly**, you have two options:

-  **Download from Google Drive:**  
  [sentiment_analysis_model.h5](https://drive.google.com/uc?id=1_ZgI6ysAOUkQtN30QjTSu7bUuiZ-sb-S)

**OR**

-  **Download `model.zip`** located in the `model/` folder of this repository and extract it.

After downloading, place the `.h5` file in the following path:
model/sentiment_analysis_model.h5

GitHub limits individual file uploads to 25MB, so the model file is hosted externally and also included as a compressed ZIP file.
Once placed correctly, you can run the project with app.py without retraining the model.

## Project Structure
- app.py: Main application file
- notebooks/: Model training and comparison notebooks
- arduino/: Code for the heart rate monitoring circuit
  > !!! If you do not have the actual Arduino hardware, you can still run the application by manually entering sample or estimated heart rate values in the web interface.
- templates/: HTML templates
- static/: CSS and image assets
- model/: Training data and model files

## Project Team Members
- İsmet Eren Coşkun
- Utku Uzunhüseyin
- Muhammet Emir Orhan
- Melih Durkun
