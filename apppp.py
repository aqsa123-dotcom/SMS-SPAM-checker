import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler
model_filename = 'best_model.pkl'  # Path to your saved model
scaler_filename = 'scaler.pkl'  # Path to your saved scaler

# Load the model
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load the scaler
with open(scaler_filename, 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app interface
st.title('Credit Card Fraud Detection App')
st.write('Enter transaction details below to check for fraud.')

# Create input fields for the features (use your dataset's feature names)
feature_names = [
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17',
    'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25',
    'V26', 'V27', 'V28', 'Amount'
]

# Create a list to store input values
input_data = []
for feature in feature_names:
    value = st.text_input(f'Enter value for {feature}:', value='0.0')
    input_data.append(float(value))

# Predict button
if st.button('Predict'):
    # Scale the input data
    input_data_scaled = scaler.transform([input_data])
    
    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Display the result
    if prediction[0] == 1:
        st.write("**Result:** Fraud detected!")
    else:
        st.write("**Result:** No fraud detected.")

# Additional information
st.write("**Note:** Please ensure you provide the values in the correct format as per the model's training data.")
