
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from operator import index
import joblib
from spacy.lang.en.stop_words import STOP_WORDS

model = joblib.load('C:/Users/giryi/Senior DataScientist/Capstone_Project_Giry/nlp_modelGiry.pkl')

stopwords = list(STOP_WORDS)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    new_review = [str(x) for x in request.form.values()]

    predictions = model.predict(new_review)[0]
    if predictions==0:
        return render_template('index.html', prediction_text='Negative')
    else:
        return render_template('index.html', prediction_text='Positive')


if __name__ == "__main__":
    app.run(debug=True)
