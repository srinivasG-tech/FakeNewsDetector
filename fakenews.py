import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title
st.title("📰 Fake News Detection App")

# Input
user_input = st.text_area("Enter News Text:")

# Predict
if st.button("Check if Fake"):
    if user_input.strip() == "":
        st.warning("Please enter some news content.")
    else:
        vectorized_input = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_input)
        confidence = model.decision_function(vectorized_input)

        # Show result
        if prediction[0] == 0:
            st.success("✅ This news is Real!")
        else:
            st.error("🚫 This news is Fake!")

        st.caption(f"Confidence Score: {round(confidence[0], 2)}")
