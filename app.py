import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask,request,app,jsonify,url_for,render_template
from pathlib import Path
from statistics import mode
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    p=[]
    for fold in range(0,5):
        regmodel=joblib.load(Path(f"./model_pkl/{fold}.pkl"))
        scalar=joblib.load(Path(f"./scaler_pkl/{fold}_scaler.pkl"))
        final_input=scalar.transform(np.array(data).reshape(1,-1))
        output=regmodel.predict(final_input)[0]
        p.append(output)
    return render_template("home.html",prediction_text="The Wine Prediction {}".format(output))



if __name__=="__main__":
    app.run(debug=True)