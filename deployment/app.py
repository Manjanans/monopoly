from flask import Flask, request, render_template, jsonify
from xgboost import XGBRegressor
import pickle
import numpy as np

# Load the model
with open("international_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("quantile_international.pkl", "rb") as f:
    quantile = pickle.load(f)
with open("quantile_predict_international.pkl", "rb") as f:
    int_predict = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")  # Serve the HTML page

@app.route("/predict", methods=["POST"])
def predict():
    # Get data from form
    feature1 = float(request.form["feature1"])
    feature2 = float(request.form["feature2"])

    # Prepare features for prediction
    features = int_predict.transform(np.array([feature1, feature2]).reshape(1, -1))

    # Make prediction
    prediction = model.predict(features)
    prediction = quantile.inverse_transform(prediction.reshape(-1, 1))

    # Return prediction to the HTML page
    return render_template("index.html", prediction=int(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)
