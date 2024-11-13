import streamlit as st
from joblib import load
import re

# Load saved model and vectorizer
model = load('sentiment_model.pkl')
vectorizer = load('tfidf_vectorizer.pkl')

# user input
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    
    # Handle specific known phrases for "positive" sentiment
    positive_phrases = ['good day', 'hello', 'how are you', 'have a nice day', 'good morning']
    if any(phrase in text for phrase in positive_phrases):
        return 'positive phrase'  # Return a dummy string to indicate a positive sentiment

    return text

# Streamlit 
st.title("Sentiment Analysis of Tweets")

# Text input for the user to enter their tweet
user_input = st.text_area("Enter your tweet:")

if st.button("Predict"):
    # Preprocess user input and convert it into the format the model expects
    processed_input = preprocess_text(user_input)
    
    if processed_input == 'positive phrase':  # If we match any of the positive phrases
        sentiment = "Positive"
    else:
        input_tfidf = vectorizer.transform([processed_input])

        # prediction using the loaded model
        prediction = model.predict(input_tfidf)

        #  result
        sentiment = "Positive" if prediction[0] == 4 else "Negative"
    
    st.write(f'The sentiment of the tweet is: {sentiment}')
