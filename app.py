# app.py

from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import joblib
import pandas as pd
import os
import uuid 

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-only-change-me')

# Load the trained pipeline

model_pipeline = joblib.load('heart_disease_model_pipeline.pkl')

# Define expected input features 
expected_features = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholestoral',
                     'fasting_blood_sugar', 'rest_ecg', 'Max_heart_rate', 'exercise_induced_angina',
                     'oldpeak', 'slope', 'vessels_colored_by_flourosopy', 'thalassemia']

@app.route('/')
def home():
    return render_template('interface.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()

        if not all(feature in data for feature in expected_features):
             return jsonify({'error': 'Missing input features'}), 400

        try:

            for col in ['age', 'resting_blood_pressure', 'cholestoral', 'Max_heart_rate', 'oldpeak']:
                if col in data:
                    data[col] = float(data[col])
        except ValueError as e:
             return jsonify({'error': f'Invalid numeric input: {str(e)}'}), 400

        input_data = pd.DataFrame([data])
        prediction = model_pipeline.predict(input_data)[0]
        probability = model_pipeline.predict_proba(input_data)[0]

        return jsonify({
            'prediction': int(prediction),
            'probability_class_0': float(probability[0]),
            'probability_class_1': float(probability[1])
        })
    except Exception as e:
        app.logger.error(f"Error during manual prediction: {e}") 
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500


@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        # 1.verify if the user uploaded a file
        if 'csv_file' not in request.files:
            flash('You did not upload any file.')
            return redirect(url_for('home'))

        file = request.files['csv_file']

        # 2.verify if the user uploaded a file
        if file.filename == '':
            flash('there is no selected file.')
            return redirect(url_for('home'))

        # 3. validate file type
        if file and file.filename.endswith('.csv'):
            # 4. save the file temporarily
            filename = f"temp_{uuid.uuid4().hex}.csv"
            filepath = os.path.join("uploads", filename)
            
            # validate if uploads directory exists, if not create it
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            file.save(filepath)

            # 5. read the file into a DataFrame
            input_df = pd.read_csv(filepath)

            # 6. validate columns
            if not all(feature in input_df.columns for feature in expected_features):
                missing_cols = set(expected_features) - set(input_df.columns)
                os.remove(filepath)
                flash(f'The file does not contain the required columns. The missing columns are: {missing_cols}')
                return redirect(url_for('home'))

            # 7. make predictions
            predictions = model_pipeline.predict(input_df)
            probabilities = model_pipeline.predict_proba(input_df)

            # 8. prepare results
            results_df = input_df.copy()
            results_df['prediction'] = predictions
            results_df['probability_of_disease'] = probabilities[:, 1]

            # 9. save results to a new CSV file
            results_filename = f"results_{uuid.uuid4().hex}.csv"
            results_filepath = os.path.join("uploads", results_filename)
            results_df.to_csv(results_filepath, index=False)

            # 10. remove the temporary uploaded file
            os.remove(filepath)

            # 11. inform the user
            flash(f'Predicted! The results have been saved in: uploads/{results_filename}')
            return redirect(url_for('home'))

        else:
            flash('Please upload a valid CSV file.')
            return redirect(url_for('home'))

    except Exception as e:
        # Clean up the uploaded file in case of error
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        app.logger.error(f"Error during CSV prediction: {e}") 
        flash(f'An error occurred: {str(e)}')
        return redirect(url_for('home'))

if __name__ == '__main__':
    # Ensure the uploads directory exists
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True) # Set debug=False in production
