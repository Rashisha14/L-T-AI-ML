# L-T-AI-ML

An AI/ML-based healthcare project that **detects bone fractures from X-ray images** and **predicts cholesterol levels** using patient data such as **blood pressure (BP), BMI, and WBC count**.

---

# <p align="center">L-T AI ML Healthcare Project</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"></a>
  <a href="#"><img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras"></a>
  <a href="#"><img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"></a>
  <a href="#"><img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"></a>
</p>

An **AI/ML healthcare project** that detects **bone fractures from X-ray images** and predicts **cholesterol levels** using patient data such as **blood pressure (BP), BMI, and WBC count**.  

---

## 📌 Table of Contents
1. [Key Features](#key-features)
2. [Installation Guide](#installation-guide)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Screenshots](#screenshots)
6. [Technologies Used](#technologies-used)
7. [License](#license)

---

## 🚀 Key Features

- 🩻 **Fracture Detection**:  
  Upload an X-ray and let the CNN model detect fractures.

- 🧪 **Cholesterol Prediction**:  
  Input health data like BP, BMI, and WBC count to estimate cholesterol levels.

- 📊 **Model Training**:  
  Includes scripts to retrain/update both models.

- 🌐 **Web Interface**:  
  Flask app for real-time predictions with simple UI.

---

## ⚙️ Installation Guide

1. **Clone the Repository**
```bash
git clone https://github.com/Rashisha14/L-T-AI-ML.git
cd L-T-AI-ML
Create Virtual Environment & Install Dependencies

bash
Copy code
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Run the Flask App

bash
Copy code
python app.py
Open http://localhost:5000 in your browser.

▶️ Usage
Fracture Detection: Upload an X-ray image → model highlights fracture presence.

Cholesterol Prediction: Enter BP, BMI, and WBC → model predicts cholesterol category/value.

📂 Project Structure
bash
Copy code
├── app.py                  # Flask app entry point
├── train1.py               # Train fracture detection model
├── train2.py               # Train cholesterol prediction model
├── models/                 # Saved ML models (.pkl / .h5)
├── data/                   # Datasets (CSV/X-ray images)
├── templates/              # HTML templates for Flask
├── static/uploads/         # Uploaded X-ray demo images
├── requirements.txt        # Dependencies
└── README.md               # Documentation
🖼 Screenshots
Fracture Detection
<p align="center"> <img src="static/uploads/fracture_demo.png" alt="Fracture Detection Demo" width="600"> </p>
Cholesterol Prediction
<p align="center"> <img src="static/uploads/cholesterol_demo.png" alt="Cholesterol Prediction Demo" width="600"> </p>
🛠 Technologies Used
<p align="left"> <a href="#"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"></a> <a href="#"><img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"></a> <a href="#"><img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white"></a> <a href="#"><img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"></a> <a href="#"><img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white"></a> </p>
Language: Python

Framework: Flask

Libraries: TensorFlow, Keras, Scikit-Learn, OpenCV, NumPy, Pandas

📜 License

MIT License

<p align="left"> <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a> </p> ```

<p align="left"> <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT 

