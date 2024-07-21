from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import pickle
import pandas as pd
from flask_cors import CORS
import numpy as np

app=Flask(__name__)
CORS(app)

with open("model","rb") as model_file:
    model = pickle.load(model_file)

@app.route("/", methods=["GET"])
def index():
    return "Hello!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features=[
        data['white blood cell count'],
        data['blood glucose random'],
        data['blood urea'],
        data['serum creatinine'],
        data['packed cell volume'],
        data['albumin'],
        data['haemoglobin'],
        data['age'],
        data['sugar'],
        data['hypertension'],
    ]
    featuresarray=np.array([features])
    prediction=model.predict(featuresarray)
    predictvalue=prediction[0].item()
    print(prediction)
    return jsonify({"prediction":predictvalue})

if __name__=="__main__":
    app.run(debug=True)