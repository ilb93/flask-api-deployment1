import os
import tensorflow as tf
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Chargement du modèle et du tokenizer
MODEL_PATH = "lstm_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    tokenizer = joblib.load(TOKENIZER_PATH)
    print("Modèle et tokenizer chargés avec succès.")
except Exception as e:
    model = None
    tokenizer = None
    app.logger.error(f"Erreur lors du chargement : {e}")

@app.route("/", methods=["GET"])
def home():
    return "L'API Flask est opérationnelle."

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not tokenizer:
        return jsonify({"error": "Modèle ou tokenizer non chargé."}), 500

    data = request.get_json()
    if not data or "tweets" not in data:
        return jsonify({"error": "Données manquantes ou mal formatées."}), 400

    tweets = data["tweets"]
    try:
        sequences = tokenizer.texts_to_sequences(tweets)
        predictions = model.predict(sequences).flatten().tolist()
        return jsonify({"predictions": predictions})
    except Exception as e:
        app.logger.error(f"Erreur lors de la prédiction : {e}")
        return jsonify({"error": "Erreur lors de la prédiction."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


