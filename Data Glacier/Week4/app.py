from flask import Flask, request, render_template
from model import lr
import pandas as pd 

# Initializing app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Getting data from form
        gender = request.form['gender']
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = request.form['ever_married']
        work_type = request.form['work_type']
        residence_type = request.form['Residence_type']
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = request.form['smoking_status']

        # Initializing feature vector of zeros for all 15 columns
        input_array = [0] * 15

        # Assigning numerical features directly
        input_array[0] = age
        input_array[1] = hypertension
        input_array[2] = heart_disease
        input_array[3] = avg_glucose_level
        input_array[4] = bmi

        # Mapping form inputs to one-hot encoded columns
        # Gender
        if gender == 'Male':
            input_array[5] = 1

        # Ever Married
        if ever_married == 'Yes':
            input_array[6] = 1

        # Work Type
        if work_type == 'Never worked':
            input_array[7] = 1
        elif work_type == 'Private':
            input_array[8] = 1
        elif work_type == 'Self-employed':
            input_array[9] = 1
        elif work_type == 'children':
            input_array[10] = 1

        # Residence Type
        if residence_type == 'Urban':
            input_array[11] = 1

        # Smoking Status
        if smoking_status == 'formerly smoked':
            input_array[12] = 1
        elif smoking_status == 'never smoked':
            input_array[13] = 1
        elif smoking_status == 'smokes':
            input_array[14] = 1

        input_data = pd.DataFrame([input_array],columns=input_array)

        # Making prediction
        prediction = lr.predict(input_data)

        # Creating a message based on the prediction result
        if prediction == 1:
            prediction_message = "This person is at risk; they may have a stroke."
        else:
            prediction_message = "Not at risk of having a stroke."

        return render_template('stroke.html', prediction=prediction_message)

    except Exception as e:
        return str(e)


@app.route('/')
def stroke_page():
    return render_template('stroke.html')


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
