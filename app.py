from flask import Flask, request, jsonify
import nltk

from flask_cors import CORS
import os
import re
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from flask import render_template

# -----------------------------------
# App Configuration
# -----------------------------------

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TOKENIZERS_DIR = os.path.join(BASE_DIR, "tokenizers")

max_length = 300

models = {
    "english": tf.keras.models.load_model(os.path.join(MODELS_DIR, "fake_news_detector_English.keras"))
}

tokenizers = {
    "english": pickle.load(open(os.path.join(TOKENIZERS_DIR, "tokenizer_English.pickle"), "rb"))
}

# -----------------------------------
# Basic Preprocessing (Optimized)
# -----------------------------------

english_stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
    'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 
    'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
])


def preprocess_text(text, language):
    text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'<.*?>', '', text)

    if language == "english":
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", '', text)
        words = word_tokenize(text)
        words = [w for w in words if w not in english_stop_words]
        return " ".join(words)

    return text

# -----------------------------------
# Homepage
# -----------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------------
# Prediction Endpoint
# -----------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "text" not in data or "language" not in data:
            return jsonify({"error": "Invalid request format"}), 400

        text = data["text"]
        language = data["language"].lower()

        if language not in models or language not in tokenizers:
            return jsonify({"error": "Unsupported language"}), 400

        token = tokenizers[language]
        model = models[language]

        processed_text = preprocess_text(text, language)

        sequence = token.texts_to_sequences([processed_text])
        padded = pad_sequences(sequence, maxlen=max_length, padding="post", truncating="post")

        probability = float(model.predict(padded, verbose=0)[0][0])

        if probability >= 0.5:
            label = "REAL NEWS"
            confidence = probability
        else:
            label = "FAKE NEWS"
            confidence = 1 - probability

        return jsonify({
            "prediction": label,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------------
# Production Entry
# -----------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
