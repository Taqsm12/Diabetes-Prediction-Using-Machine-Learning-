from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import logging

app = Flask(__name__, template_folder='template')
logging.basicConfig(level=logging.INFO)

# Load the SVM model
try:
    svm_model = pickle.load(open('svm_model.pkl', 'rb'))
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error("Error loading the model: %s", e)
    svm_model = None

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/send', methods=['POST'])
def getdata():
    try:
        # Extract features from the form
        features = [float(x) for x in request.form.values()]
        final_features = np.array([features])
        
        # Predict using the loaded SVM model
        prediction = svm_model.predict(final_features)
        logging.info("Prediction: %s", prediction)

        # Determine result based on prediction
        if prediction[0] == 0:
            result = "You Are Non-Diabetic"
        else:
            result = "You Are Diabetic"

        # Extract individual form values
        Pregnancies = request.form['Pregnancies']
        Glucose = request.form['Glucose']
        BloodPressure = request.form['BloodPressure']
        SkinThickness = request.form['SkinThickness']
        Insulin = request.form['Insulin']
        BMI = request.form['BMI']
        DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
        Age = request.form['Age']
        
        # Render result page
        return render_template('show.html', preg=Pregnancies, bp=BloodPressure,
                               gluc=Glucose, st=SkinThickness, ins=Insulin, bmi=BMI,
                               dbf=DiabetesPedigreeFunction, age=Age, res=result)
    except Exception as e:
        logging.error("Error in getdata: %s", e)
        return "An error occurred during prediction."

if __name__ == "__main__":
    app.run(debug=True)
