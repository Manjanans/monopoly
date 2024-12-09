from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the model
with open("num_acc_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("quantile_num_acc.pkl", "rb") as f:
    quantile = pickle.load(f)
with open("quantile_predict_num_acc.pkl", "rb") as f:
    int_predict = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("predict_num_acc.html")  # Serve the HTML page

@app.route("/predict", methods=["POST"])
def predict():
    # Get data from form
    feature1 = float(request.form["feature1"])
    feature2 = float(request.form["feature2"])
    feature3 = float(request.form["feature3"])

    # Prepare features for prediction
    features = int_predict.transform(np.array([feature1, feature2, feature3]).reshape(1, -1))

    # Make prediction
    pred = model.predict(features)
    pred = quantile.inverse_transform(pred.reshape(-1, 1))

    texto = 'Tiene 1 cuenta' if pred[0] == 0 else 'Tiene 2 o máscuentas'

    # Return prediction to the HTML page
    return render_template("predict_num_acc.html", prediction=texto)

if __name__ == "__main__":
    app.run(debug=True)
