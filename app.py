import os
from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np
import logging

# Initialiser l'application Flask
app = Flask(__name__)

# Configurer les logs pour Heroku
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger le modèle LSTM
try:
    model = tf.keras.models.load_model("lstm_model.keras")
    logger.info("Modèle LSTM chargé avec succès.")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle LSTM : {e}")
    model = None

# Charger le tokenizer
try:
    tokenizer = joblib.load("tokenizer.pkl")
    logger.info("Tokenizer chargé avec succès.")
except Exception as e:
    logger.error(f"Erreur lors du chargement du tokenizer : {e}")
    tokenizer = None

@app.route("/")
def home():
    return "Bienvenue sur mon API Flask déployée sur Heroku !"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        tweets = data.get('tweets', [])
        if not tweets:
            return jsonify({"error": "Aucun tweet fourni."}), 400

        if model is None or tokenizer is None:
            return jsonify({"error": "Modèle ou tokenizer non chargé."}), 500

        # Préparer les tweets pour le modèle
        sequences = tokenizer.texts_to_sequences(tweets)
        max_length = 100
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length)

        # Faire les prédictions
        predictions = model.predict(padded_sequences)

        # Formater les résultats
        results = [
            {"tweet": tweet, "sentiment": "positif" if prediction > 0.5 else "négatif"}
            for tweet, prediction in zip(tweets, predictions.flatten())
        ]

        return jsonify({"predictions": results})
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
