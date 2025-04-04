from flask import Flask, request, jsonify
import numpy as np
from trainedmodel import train_model  # Import the ensemble model

app = Flask(__name__)

# Load the trained ensemble model
model = train_model()

@app.route("/")
def home():
    return "Diabetes Prediction API with Ensemble Learning is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Receive JSON input
    features = np.array([
        data["Glucose"], data["BloodPressure"], data["SkinThickness"],
        data["Insulin"], data["BMI"], data["DiabetesPedigreeFunction"], data["Age"]
    ]).reshape(1, -1)

    prediction = model.predict(features)[0]
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)