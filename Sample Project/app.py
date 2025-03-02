from flask import Flask, request, jsonify
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load trained model
with open("spam_classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocessing function
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = " ".join([stemmer.stem(word) for word in text.split() if word not in stop_words])  # Remove stopwords & stem
    return text

# Define API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Receive JSON input
        email_text = data.get("email_text", "")  # Extract email text

        if not email_text:
            return jsonify({"error": "Email text is required"}), 400

        # Clean and transform input text
        cleaned_text = clean_text(email_text)
        transformed_text = vectorizer.transform([cleaned_text])

        # Predict using the model
        prediction = model.predict(transformed_text)[0]
        result = "spam" if prediction == 1 else "safe"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
