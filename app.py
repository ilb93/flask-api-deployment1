import os
from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure.trace_exporter import AzureExporter
from opencensus.trace import integrations
from opencensus.trace.samplers import ProbabilitySampler
from opencensus.trace.tracer import Tracer

# Initialiser l'application Flask
app = Flask(__name__)

# Configuration Azure Application Insights
INSTRUMENTATION_KEY = "47019b65-b8ca-40be-95c8-a0552c3b62b3"  # Clé d'instrumentation réelle
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(AzureLogHandler(connection_string=f"InstrumentationKey={INSTRUMENTATION_KEY}"))

# Initialiser le traceur OpenCensus pour les traces
tracer = Tracer(exporter=AzureExporter(connection_string=f"InstrumentationKey={INSTRUMENTATION_KEY}"),
                sampler=ProbabilitySampler(1.0))

# Ajouter des intégrations pour la journalisation et les requêtes
integrations.add_integrations(['logging', 'requests'])

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
    """Route d'accueil."""
    return "Bienvenue sur mon API Flask déployée sur Heroku avec Azure !"

@app.route("/predict", methods=["POST"])
def predict():
    """Route pour faire des prédictions de sentiments."""
    try:
        data = request.get_json()
        tweets = data.get('tweets', [])
        if not tweets:
            return jsonify({"error": "Aucun tweet fourni."}), 400

        # Transformer les tweets en séquences
        sequences = tokenizer.texts_to_sequences(tweets)
        max_length = 100  # Ajustez selon les besoins du modèle
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length)

        # Faire des prédictions
        predictions = model.predict(padded_sequences)

        # Formater les résultats
        results = [
            {
                "tweet": tweet,
                "sentiment": "positif" if prediction > 0.5 else "négatif"
            }
            for tweet, prediction in zip(tweets, predictions.flatten())
        ]

        return jsonify({"predictions": results})
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Utiliser le port attribué par Heroku ou 5000 par défaut
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

