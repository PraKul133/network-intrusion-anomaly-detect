import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load and prepare the data
@st.cache_data
def load_data():
    # Load your data
    data = pd.read_csv('Train_data.csv')  # Replace with the correct path to your CSV
    return data

# Train the Random Forest model
@st.cache_data
def train_model(data):
    # Select integer columns for features
    X = data[['duration', 'src_bytes', 'dst_bytes', 'num_failed_logins', 'count', 'srv_count']]
    y = data['class'].apply(lambda x: 1 if x == 'anomaly' else 0)  # Target: 1 for anomaly, 0 for normal
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Test the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

# Predict with the model
def predict_intrusion(model, user_input):
    # Convert input data to numpy array
    input_data_as_numpy_array = np.asarray(user_input)
    
    # Reshape the array for prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_data_reshaped)
    
    if prediction[0] == 0:
        return 'Normal'
    else:
        return 'Anomaly'

def main():
    st.title('Network Intrusion Detection System')

    # Load data
    data = load_data()
    
    # Train model
    model, accuracy = train_model(data)
    
    st.write(f"Model Training Accuracy: {accuracy:.2f}")
    
    # User input features
    st.sidebar.subheader('User Input Parameters')
    
    duration = st.sidebar.slider('Duration', 0, 100, 0)
    src_bytes = st.sidebar.slider('Source Bytes', 0, 100, 0)
    dst_bytes = st.sidebar.slider('Destination Bytes', 0, 100, 0)
    num_failed_logins = st.sidebar.slider('Number of Failed Logins', 0, 10, 0)
    count = st.sidebar.slider('Count', 0, 100, 0)
    srv_count = st.sidebar.slider('Service Count', 0, 100, 0)

    # Collect user input into a list
    user_input = [duration, src_bytes, dst_bytes, num_failed_logins, count, srv_count]

    # Prediction
    diagnosis = ''
    if st.button('Test Network'):
        diagnosis = predict_intrusion(model, user_input)
    
    st.success(f'Prediction: {diagnosis}')

if __name__ == '__main__':
    main()
