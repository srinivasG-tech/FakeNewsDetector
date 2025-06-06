# 📰 Fake News Detector

A simple machine learning web application to detect fake news using **Scikit-learn**, **Streamlit**, and **TF-IDF vectorization**.

## 🔍 Features
- Detect whether a news article is fake or real
- Uses PassiveAggressiveClassifier
- Trained on `Fake.csv` and `True.csv` datasets
- Real-time input via web interface

## 🚀 How to Run Locally

```bash
git clone https://github.com/srinivasG-tech/FakeNewsDetector.git
cd FakeNewsDetector
pip install -r requirements.txt
streamlit run fakenews.py
