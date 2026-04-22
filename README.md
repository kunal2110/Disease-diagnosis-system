🏥 Disease Prediction System (Flask + Machine Learning)

A multi-disease prediction web application built using Machine Learning and Flask, capable of predicting:

🧬 Diabetes
❤️ Heart Disease
🧠 Parkinson’s Disease
🧪 Liver Disease

This project provides a user-friendly web interface where users can input medical parameters and get instant predictions.

🚀 Features
🔍 Predict multiple diseases in one platform
🌐 Flask-based web application
🎨 Modern UI (Black + Blue Theme)
⚡ Fast and real-time predictions
🧠 Machine Learning models integrated
📱 Responsive and clean design
🛠️ Tech Stack
Frontend: HTML, CSS
Backend: Flask (Python)
Machine Learning: Scikit-learn, NumPy, Pandas
Model Storage: Pickle (.sav files)
📁 Project Structure
project/
│
├── app.py                  # Flask backend
├── models/                # Saved ML models
│   ├── diabetes_model.sav
│   ├── heart_model.sav
│   ├── parkinson_model.sav
│   └── liver_model.sav
│
├── templates/
│   └── index.html         # Frontend UI
│
├── static/
│   └── style.css          # Styling
│
└── README.md
⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/disease-prediction-system.git
cd disease-prediction-system
2️⃣ Install Dependencies
pip install -r requirements.txt

If requirements.txt not available:

pip install flask numpy pandas scikit-learn
3️⃣ Run the Application
python app.py
4️⃣ Open in Browser
http://127.0.0.1:5000/

🧠 Machine Learning Models

Each disease prediction uses a trained ML model:

Diabetes → Logistic Regression / SVM
Heart Disease → Classification Model
Parkinson → Random Forest / SVM
Liver Disease → Classification Model
⚠️ Disclaimer

This application is for educational purposes only and should not be used as a substitute for professional medical advice.
