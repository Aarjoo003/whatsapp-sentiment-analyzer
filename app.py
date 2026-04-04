from flask import Flask, render_template, request
import pandas as pd
import re
import nltk
import os
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# -------------------------------
# ✅ Load NLTK stopwords safely
# -------------------------------
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# -------------------------------
# ✅ Text Cleaning Function
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# -------------------------------
# ✅ Load or Train Model
# -------------------------------
MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
    print("✅ Loading saved model...")
    model = pickle.load(open(MODEL_FILE, "rb"))
    vectorizer = pickle.load(open(VECTORIZER_FILE, "rb"))

else:
    print("⚡ Training model (first time only)...")

    df = pd.read_csv("whatsapp_reviews.csv")

    # Remove neutral reviews
    df = df[df['rating'] != 3]

    # Convert rating to sentiment
    df['sentiment'] = df['rating'].apply(lambda r: 1 if r > 3 else 0)

    # Clean text
    df['clean_text'] = df['review_text'].apply(clean_text)

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(df['clean_text'])
    y = df['sentiment']

    # Train Model
    model = LogisticRegression(C=1, solver='lbfgs', max_iter=1000)
    model.fit(X, y)

    # Save model
    pickle.dump(model, open(MODEL_FILE, "wb"))
    pickle.dump(vectorizer, open(VECTORIZER_FILE, "wb"))

    print("✅ Model trained & saved!")

# -------------------------------
# ✅ Routes
# -------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    user_input = ""

    if request.method == 'POST':
        user_input = request.form.get('review', '')

        if user_input.strip():
            cleaned = clean_text(user_input)
            vectorized = vectorizer.transform([cleaned])

            pred = model.predict(vectorized)[0]
            proba = model.predict_proba(vectorized)[0]

            confidence = round(max(proba) * 100, 2)
            prediction = "Positive" if pred == 1 else "Negative"

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        user_input=user_input
    )

# -------------------------------
# ✅ Run App
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)