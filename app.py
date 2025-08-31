from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

# Load vectorizer and models
vectorizer = joblib.load("models/vectorizer.pkl")
default_model = joblib.load("models/model.pkl")
models = {
    "Logistic Regression": joblib.load("models/logistic_model.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes.pkl"),
}

# Set individual model accuracies
accuracies = {
    "Logistic Regression": 95.8,
    "Random Forest": 94.2,
    "Naive Bayes": 90.5,
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    news = ""
    final_prediction = ""
    model_outputs = {}

    if request.method == "POST":
        news = request.form.get("news", "").strip()

        if news:
            cleaned = clean_text(news)
            vectorized = vectorizer.transform([cleaned])

            for name, model in models.items():
                pred = model.predict(vectorized)[0]
                model_outputs[name] = "REAL" if pred == 1 else "FAKE"

            final_prediction = "REAL" if default_model.predict(vectorized)[0] == 1 else "FAKE"
        else:
            final_prediction = "Please enter some news text."

    return render_template("result.html",
                           news=news,
                           prediction=final_prediction,
                           outputs=model_outputs,
                           accuracies=accuracies)

if __name__ == "__main__":
    app.run(debug=True)
