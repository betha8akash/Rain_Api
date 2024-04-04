from flask import Flask, request, jsonify
import pickle
import pandas as pd
import joblib

model = joblib.load(open('ANDHRA PRADESH_sarima_model.pkl', 'rb'))

model1 = joblib.load(open('RAYALSEEMA_sarima_model.pkl', 'rb'))

model2 = joblib.load(open('TELANGANA_sarima_model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "Hello world"


@app.route('/andhra')
def andhra():
    result = sum(model.forecast(steps=6))
    result2 = sum(model2.forecast(steps=6))
    result1 = sum(model1.forecast(steps=6))
    return jsonify({'Andhra': result, 'Rayalaseema': result1, 'Telangana': result2})


if __name__ == '__main__':
    app.run(debug=True)
