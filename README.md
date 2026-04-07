🚀 WhatsApp Sentiment Analyzer

An AI-powered web application that analyzes WhatsApp Play Store reviews and predicts whether the sentiment is **Positive 😊** or **Negative 😞** using Machine Learning.

🌐 Live Demo

👉https://huggingface.co/spaces/aarzoodahiya81/Whatsapp_sentiment-analysis

📌 Project Overview

This project uses Natural Language Processing (NLP) techniques to process and analyze user reviews.
It converts raw text into numerical features using TF-IDF and applies a Logistic Regression model to classify sentiment.

🧠 Tech Stack

* **Backend:** Python, Flask
* **Machine Learning:** Scikit-learn (Logistic Regression)
* **NLP:** TF-IDF Vectorization, Stopword Removal
* **Frontend:** HTML, CSS, JavaScript
* **Deployment:** Render (Cloud Platform)

✨ Features

* 🔍 Real-time sentiment prediction
* 📊 Confidence score display
* 🎨 Beautiful interactive UI
* 🧠 Token-based sentiment highlighting
* 📜 Analysis history tracking
* ⚡ Fast and responsive performance

⚙️ How It Works

1. User inputs a review
2. Text is preprocessed (cleaning + stopword removal)
3. TF-IDF converts text into numerical vectors
4. Logistic Regression model predicts sentiment
5. Output displayed with confidence score

📊 Model Details

* **Algorithm:** Logistic Regression
* **Accuracy:** 77.67%
* **Vectorization:** TF-IDF (15,000 features)
* **N-grams:** Unigrams + Bigrams
* **Dataset Size:** ~5,400 reviews
* **Classes:** Binary (Positive / Negative)

📁 Project Structure

whatsapp-sentiment-analyzer/
│
├── app.py
├── requirements.txt
├── Procfile
├── whatsapp_reviews.csv
│
└── templates/
    └── index.html

🚀 Installation & Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/whatsapp-sentiment-analyzer.git

# Navigate to project folder
cd whatsapp-sentiment-analyzer

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

🎯 Use Cases

* Customer feedback analysis
* App review monitoring
* Sentiment tracking for businesses
* Social media analytics

🔥 Future Improvements

* 🔹 Add Deep Learning models (LSTM / BERT)
* 🔹 Improve accuracy to 85%+
* 🔹 Multi-language support
* 🔹 Mobile app integration

👨‍💻 Author
**Aarjoo Dahiya**
B.Tech Third Year Student 

⭐ Show Your Support

If you like this project, give it a ⭐ on GitHub!

