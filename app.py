import numpy as np
import pickle
from flask import Flask, render_template, request

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('page.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [
            float(request.form['radius_mean']),
            float(request.form['texture_mean']),
            float(request.form['perimeter_mean']),
            float(request.form['area_mean']),
            float(request.form['concave_points_mean']),
            float(request.form['radius_worst']),
            float(request.form['texture_worst']),
            float(request.form['perimeter_worst']),
            float(request.form['area_worst']),
            float(request.form['concave_points_worst'])
        ]
        
        input_data = np.array(input_data).reshape(1, -1)

        prediction = model.predict(input_data)

        result = 'M A L I G N A N T' if prediction[0] == 1 else 'B E N I G N'

        print(prediction[0])

        return render_template('page.html', prediction=result)
        
    except Exception as e:
        print(f"Error: {e}")
        return render_template('page.html', prediction="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
