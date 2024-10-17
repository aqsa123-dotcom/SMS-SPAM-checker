import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Download NLTK data if not present
nltk.download('punkt')
nltk.download('stopwords')

# Load the saved TfidfVectorizer and model
vectorizer_filename = 'tfidf_vectorizer.pkl'
model_filename = 'best_model.pkl'

with open(vectorizer_filename, 'rb') as file:
    tfidf = pickle.load(file)

with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Function to transform and predict
def predict_spam(text):
    transformed_text = tfidf.transform([text]).toarray()
    prediction = model.predict(transformed_text)
    return 'ham' if prediction[0] == 0 else 'spam'

# Streamlit app interface
st.title('SMS Spam Detection App')
st.write('Enter the SMS message below and click "Predict" to find out if it is ham or spam.')

# Input field for the SMS message
input_message = st.text_area("Enter SMS text:")

# Prediction button
if st.button('Predict'):
    if input_message.strip() != '':
        result = predict_spam(input_message)
        st.write(f"The message is classified as: **{result.upper()}**")
    else:
        st.write("Please enter a valid SMS message.")
