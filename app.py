import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('models/iris_model.pkl')

st.title("🌸 Iris Flower Species Prediction 🌸")

# Input fields for flower measurements
features = [
    st.number_input("Sepal Length (cm)", min_value=0.0),
    st.number_input("Sepal Width (cm)", min_value=0.0),
    st.number_input("Petal Length (cm)", min_value=0.0),
    st.number_input("Petal Width (cm)", min_value=0.0)
]

if st.button("Predict"):
    prediction = model.predict([features])
    species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    st.success(f"Predicted Species: {species[prediction[0]]}")
