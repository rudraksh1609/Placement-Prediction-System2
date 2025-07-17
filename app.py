from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler using joblib
model = joblib.load('placement_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature keys expected from frontend JSON (match names in HTML)
feature_keys = {
    'cgpa': 'CGPA',
    'major': 'Major Projects',
    'certs': 'Workshops/Certificatios',
    'mini': 'Mini Projects',
    'skills': 'Skills',
    'comm': 'Communication Skill Rating',
    'Internship': 'Internship',
    'Hackathon': 'Hackathon',
    'perc_12': '12th Percentage',
    'perc_10': '10th Percentage',
    'backlogs': 'backlogs'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Reorder and map the input as per model's expectation
        feature_values = []
        for html_key, model_key in feature_keys.items():
            value = data.get(html_key)
            if model_key in ['Internship', 'Hackathon']:
                feature_values.append(int(value))
            else:
                feature_values.append(float(value))

        # Reshape and scale input
        input_array = np.array([feature_values])
        input_df = pd.DataFrame([feature_values], columns=feature_keys.values())
        scaled_input = scaler.transform(input_df)

        # Predict
        prediction = model.predict(scaled_input)[0]
        result = "🎯 Placed!" if prediction == 1 else "❌ Not Placed"

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'result': f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
