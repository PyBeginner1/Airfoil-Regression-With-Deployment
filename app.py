import pickle
import flask
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app =Flask(__name__)

#Load Pickle file
model  = pickle.load(open('random_forest.pkl', 'rb'))

@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')

#Create an API for postman
@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    new_data = np.array(list(data.values())).reshape(1,-1)
    prediction = model.predict(new_data)[0]

    return jsonify(f'The pressure is {round(prediction,5)}')

#Create API for webpage
@app.route('/predict', methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    print(data)
    new_data = np.array(data).reshape(1,-1)
    prediction = model.predict(new_data)[0]
    print(prediction)

    return render_template('home.html', prediction_text = 'Airfoil pressure is {}'.format(prediction))


if __name__ == '__main__':
    app.run(debug = True)