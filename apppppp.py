streamlit
pandas
numpy
scikit-learn
nltk
pickle5  # If you're using pickle5; otherwise, remove this line

# Import necessary libraries
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

# Ensure NLTK dependencies are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the saved TfidfVectorizer and model
vectorizer_filename = 'tfidf_vectorizer.pkl'  # Replace with your vectorizer file path
model_filename = 'spam_classifier_model.pkl'  # Replace with your model file path

# Load the vectorizer
try:
    with open(vectorizer_filename, 'rb') as file:
        tfidf = pickle.load(file)
except FileNotFoundError:
    st.error(f"Error: {vectorizer_filename} not found. Please upload the file.")
    st.stop()

# Load the model
try:
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Error: {model_filename} not found. Please upload the file.")
    st.stop()

# Function to clean and transform the input text
def transform_text(text):
    # Lowercase the text
    text = text.lower()

    # Tokenize the text
    words = nltk.word_tokenize(text)

    # Remove stopwords and punctuation
    clean_words = [word for word in words if word.isalnum() and word not in stopwords.words('english')]

    # Return the cleaned text as a string
    return ' '.join(clean_words)

# Function to predict if the SMS is spam or ham
def predict_spam(text):
    # Transform the input text using TfidfVectorizer
    transformed_text = tfidf.transform([transform_text(text)]).toarray()

    # Predict using the loaded model
    prediction = model.predict(transformed_text)

    # Return the result as 'ham' or 'spam'
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
