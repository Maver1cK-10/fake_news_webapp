from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Dummy training data
X_train = ["this is real news", "totally fake news", "this is true", "this is false", "not trustworthy", "trustworthy news"]
y_train = [1, 0, 1, 0, 0, 1]  # 1 = REAL, 0 = FAKE

# Vectorizer
vectorizer = CountVectorizer()
X_vect = vectorizer.fit_transform(X_train)

# Train models
log_model = LogisticRegression()
rf_model = RandomForestClassifier()
nb_model = MultinomialNB()

log_model.fit(X_vect, y_train)
rf_model.fit(X_vect, y_train)
nb_model.fit(X_vect, y_train)

# Save vectorizer and models
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(log_model, "models/logistic_model.pkl")
joblib.dump(rf_model, "models/random_forest.pkl")
joblib.dump(nb_model, "models/naive_bayes.pkl")

# You can also overwrite model.pkl with logistic_model
joblib.dump(log_model, "models/model.pkl")  # Default used in your app.py
print("✅ Models and vectorizer saved successfully!")
