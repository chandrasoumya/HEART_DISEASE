from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the saved model, scaler, and encoder
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

# Define feature lists
numerical_features = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
categorical_features = ['sex', 'chest pain type', 'fasting blood sugar', 
                       'resting ecg', 'exercise angina', 'ST slope']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data from form
        input_data = {
            'age': float(request.form['age']),
            'sex': int(request.form['sex']),
            'chest pain type': int(request.form['chest_pain_type']),
            'resting bp s': float(request.form['resting_bp_s']),
            'cholesterol': float(request.form['cholesterol']),
            'fasting blood sugar': int(request.form['fasting_blood_sugar']),
            'resting ecg': int(request.form['resting_ecg']),
            'max heart rate': float(request.form['max_heart_rate']),
            'exercise angina': int(request.form['exercise_angina']),
            'oldpeak': float(request.form['oldpeak']),
            'ST slope': int(request.form['st_slope'])
        }

        # Create DataFrame for numerical and categorical features
        numerical_data = pd.DataFrame([[input_data[feat] for feat in numerical_features]], 
                                     columns=numerical_features)
        categorical_data = pd.DataFrame([[input_data[feat] for feat in categorical_features]], 
                                       columns=categorical_features)

        # Preprocess numerical features
        numerical_scaled = scaler.transform(numerical_data)
        numerical_scaled_df = pd.DataFrame(numerical_scaled, columns=numerical_features)

        # Preprocess categorical features
        categorical_encoded = encoder.transform(categorical_data)
        categorical_encoded_df = pd.DataFrame(categorical_encoded, 
                                            columns=encoder.get_feature_names_out(categorical_features))

        # Combine features
        input_processed = pd.concat([numerical_scaled_df, categorical_encoded_df], axis=1)

        # Make prediction
        prediction = model.predict(input_processed)[0]
        prediction_proba = model.predict_proba(input_processed)[0][1]

        # Interpret result
        result = 'Heart Disease' if prediction == 1 else 'Normal'
        confidence = f"{prediction_proba:.2%}" if prediction == 1 else f"{1 - prediction_proba:.2%}"

        return render_template('result.html', result=result, confidence=confidence)
    
    except Exception as e:
        return render_template('result.html', result=f"Error: {str(e)}", confidence="N/A")

if __name__ == '__main__':
    app.run(debug=True)