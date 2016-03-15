from flask import Flask, request, render_template
import cPickle as pickle
import unirest
from tweet_parse import *


WP = WordPredictor()
WP.fit()

app = Flask(__name__)
# load model
# with open('data/vectorizer.pkl') as f:
#     vectorizer = pickle.load(f)
# with open('data/model.pkl') as f:
#     model = pickle.load(f)
#

# home page
@app.route('/')
def index():


    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    text = str(request.form['precede_words'])

    prediction = WP.predict(text)

    # print 'prediction type:' ,type(prediction)
    return render_template('index.html', preceding_wd = text ,prediction = prediction, section = "prediction")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
