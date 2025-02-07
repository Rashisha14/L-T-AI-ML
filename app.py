from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load models with exception handling
try:
    linear_model = joblib.load('linear_model.pkl')
    kmeans_model = joblib.load('kmeans_model.pkl')
    fracture_model = load_model('models/fracture_model.keras')  # Load fracture model
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)  # Exit if models cannot be loaded

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model_type = request.form.get('model_type')

    if model_type == 'linear_regression':
        try:
            # Get data from the form and validate it
            age = request.form.get('age', '')
            bmi = request.form.get('bmi', '')
            blood_pressure = request.form.get('blood_pressure', '')

            # Convert to float if values exist
            if age and bmi and blood_pressure:
                input_data = np.array([[float(age), float(bmi), float(blood_pressure)]])
                prediction = linear_model.predict(input_data)[0]
            else:
                return render_template('index.html', error="Please enter valid input values.", 
                                       model_type=model_type, age=age, bmi=bmi, blood_pressure=blood_pressure)

            return render_template('index.html', model_type=model_type, prediction=prediction, 
                                   age=age, bmi=bmi, blood_pressure=blood_pressure)

        except ValueError as e:
            print(f"Error with input data for linear regression: {e}")
            return render_template('index.html', error="Invalid input data for Linear Regression",
                                   model_type=model_type, age=age, bmi=bmi, blood_pressure=blood_pressure)

    elif model_type == 'k_means':
        try:
            # Get data from the form
            age = request.form.get('age', '')
            bmi = request.form.get('bmi', '')
            treatment_success_rate = request.form.get('treatment_success_rate', '')

            # Convert to float if values exist
            if age and bmi and treatment_success_rate:
                input_data = np.array([[float(age), float(bmi), float(treatment_success_rate)]])
                cluster = kmeans_model.predict(input_data)[0]
            else:
                return render_template('index.html', error="Please enter valid input values.",
                                       model_type=model_type, age=age, bmi=bmi, treatment_success_rate=treatment_success_rate)

            return render_template('index.html', model_type=model_type, cluster=cluster,
                                   age=age, bmi=bmi, treatment_success_rate=treatment_success_rate)

        except ValueError as e:
            print(f"Error with input data for K-Means: {e}")
            return render_template('index.html', error="Invalid input data for K-Means",
                                   model_type=model_type, age=age, bmi=bmi, treatment_success_rate=treatment_success_rate)

    elif model_type == 'fracture_prediction':
        try:
            # Get the uploaded image
            img_file = request.files.get('bone_image')

            if not img_file or img_file.filename == '':
                return render_template('index.html', error="No image uploaded for Fracture Prediction.", model_type=model_type)

            img_path = os.path.join('uploads', img_file.filename)

            # Ensure the 'uploads' directory exists
            if not os.path.exists('uploads'):
                os.makedirs('uploads')

            # Save the image to the server
            img_file.save(img_path)
            print(f"Image saved at: {img_path}")

            # Preprocess image for prediction
            img = image.load_img(img_path, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = img_array / 255.0  # Rescale image

            print(f"Image array shape: {img_array.shape}, type: {img_array.dtype}")

            # Predict using the fracture model
            prediction = fracture_model.predict(img_array)[0]
            print(f"Model prediction: {prediction}")

            result = "Fracture Detected" if prediction >= 0.5 else "No Fracture"

            return render_template('index.html', model_type=model_type, fracture_result=result, bone_image=img_file.filename)

        except Exception as e:
            print(f"Error with image processing or prediction: {e}")
            return render_template('index.html', error="Error with image processing or prediction", model_type=model_type)

    return render_template('index.html', error="Invalid Model Type")

if __name__ == "__main__":
    app.run(debug=True)
