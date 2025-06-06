# 📰 Fake News Detector

A simple machine learning web application to detect fake news using **Scikit-learn**, **Streamlit**, and **TF-IDF vectorization**.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fakenewsdetector-e2e2xmpbqjsthmlvh8antc.streamlit.app)

---

## 🔍 Features

- Detect whether a news article is Fake or Real
- Uses PassiveAggressiveClassifier
- Trained on `Fake.csv` and `True.csv` datasets
- Real-time input via web interface (built with Streamlit)

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/srinivasG-tech/FakeNewsDetector.git
cd FakeNewsDetector
pip install -r requirements.txt
streamlit run fakenews.py

🧠 Model Details
Vectorization: TF-IDF

Classifier: PassiveAggressiveClassifier

Trained using train_model.py

Saved model: model.pkl

Saved vectorizer: vectorizer.pkl

📁 Project Structure

FakeNewsDetector/
├── fakenews.py           # Main Streamlit App
├── train_model.py        # Model training script
├── model.pkl             # Trained model
├── vectorizer.pkl        # TF-IDF vectorizer
├── Fake.csv              # Fake news data
├── True.csv              # Real news data
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation

🙌 Author
Srinivas G-Tech
GitHub: @srinivasG-tech

⭐ Star this repo if you found it useful!











