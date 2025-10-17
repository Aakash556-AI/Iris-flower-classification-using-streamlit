# prediction.py
import joblib
import numpy as np

# Load trained model
model = joblib.load("iris_rf_model.pkl")

# Map numeric labels to species names and optional image links
target_names = {
    0: {"name": "Iris-setosa", 
        "image": "/Users/akashkumarsinha/Desktop/iris_project/image/Kosaciec_szczecinkowaty_Iris_setosa.jpg"},
    1: {"name": "Iris-versicolor", 
        "image": "/Users/akashkumarsinha/Desktop/iris_project/image/Iris_versicolor_3.jpg"},
    2: {"name": "Iris-virginica", 
        "image": "/Users/akashkumarsinha/Desktop/iris_project/image/Iris_versicolor_3.jpg"}
}

def predict(features: np.ndarray):
    """
    Predict Iris species from features
    Returns: predicted flower info (name + image) and probabilities
    """
    pred = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    flower = target_names[pred]
    return flower, probabilities
