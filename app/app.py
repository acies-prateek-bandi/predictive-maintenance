from flask import Flask, render_template, request
import pandas as pd  # For data preprocessing
# from joblib import load  # For loading the trained model
from pickle import load

app = Flask(__name__)

# Load your trained predictive maintenance model
model = load(open('./data/06_models/binary_classifier.pickle','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    air_temp = float(request.form['air_temp'])
    process_temp = float(request.form['process_temp'])
    speed = float(request.form['speed'])
    torque = float(request.form['torque'])
    tool_wear = float(request.form['tool_wear'])

    # Prepare user input as a DataFrame (adjust according to your dataset)
    user_input = pd.DataFrame([[air_temp, process_temp, speed, torque, tool_wear]],
                              columns=['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])

    # Make predictions
    prediction = model.predict(user_input)[0]

    # Render the result page with the prediction
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
