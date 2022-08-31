import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flase app
app = Flask(__name__)

# Load the pickle model
model = pickle.load(open("model_Rant.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("template.html")

@app.route("/predcit", methods=["POST"])
def predict():
    float_featuers = [float(x) for x in request.form.values()]
    features = [np.array(float_featuers)]
    prediction = model.predict(features)

    return render_template("template.html", prediction_text = "The salary is {}".format(prediction))

if __name__== "__main__":
    app.run(debug=True)



