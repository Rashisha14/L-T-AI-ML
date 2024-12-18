from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load the models with exception handling
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
            age = float(request.form['age'])
            bmi = float(request.form['bmi'])
            blood_pressure = float(request.form['blood_pressure'])

            # Prepare input data for prediction
            input_data = np.array([[age, bmi, blood_pressure]])
            prediction = linear_model.predict(input_data)[0]
            return render_template('index.html', model_type=model_type, prediction=prediction)

        except ValueError as e:
            print(f"Error with input data for linear regression: {e}")
            return render_template('index.html', error="Invalid input data for Linear Regression")
    
    elif model_type == 'k_means':
        try:
            # Get data from the form and validate it
            age = float(request.form['age'])
            bmi = float(request.form['bmi'])
            treatment_success_rate = float(request.form['treatment_success_rate'])

            # Prepare input data for clustering
            input_data = np.array([[age, bmi, treatment_success_rate]])
            cluster = kmeans_model.predict(input_data)[0]
            return render_template('index.html', model_type=model_type, cluster=cluster)

        except ValueError as e:
            print(f"Error with input data for K-Means: {e}")
            return render_template('index.html', error="Invalid input data for K-Means")

    elif model_type == 'fracture_prediction':
        try:
            # Get the uploaded image
            img_file = request.files['bone_image']
            img_path = os.path.join('uploads', img_file.filename)

            # Ensure the 'uploads' directory exists
            if not os.path.exists('uploads'):
                os.makedirs('uploads')

            # Save the image to the server
            img_file.save(img_path)

            # Debugging: check if the image is saved correctly
            print(f"Image saved at: {img_path}")

            # Preprocess image for prediction
            img = image.load_img(img_path, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = img_array / 255.0  # Rescale image

            # Debugging: check image array shape and type
            print(f"Image array shape: {img_array.shape}, type: {img_array.dtype}")

            # Predict using the fracture model
            prediction = fracture_model.predict(img_array)[0]
            print(f"Model prediction: {prediction}")

            # Debugging: log threshold decision
            if prediction >= 0.5:
                result = "Fracture Detected"
            else:
                result = "No Fracture"
            return render_template('index.html', model_type=model_type, result=result)

        except Exception as e:
            print(f"Error with image processing or prediction: {e}")
            return render_template('index.html', error="Error with image processing or prediction")

    return render_template('index.html', error="Invalid Model Type")

if __name__ == "__main__":
    app.run(debug=True)
