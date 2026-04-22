from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# ================= LOAD MODELS =================
diabetes_model = pickle.load(open('models/diabetes_model.pkl', 'rb'))
heart_model = pickle.load(open('models/heart_model.pkl', 'rb'))
parkinson_model = pickle.load(open('models/parkinson_model.pkl', 'rb'))
liver_model = pickle.load(open('models/liver_model.pkl', 'rb'))  # ✅ NEW

# ================= COMMON FUNCTION =================
def predict(model, data):
    data = np.asarray(data).reshape(1, -1)
    return model.predict(data)[0]

# ================= DIABETES =================
@app.route('/', methods=['GET', 'POST'])
def diabetes():
    fields = ["Pregnancies", "Glucose", "BP", "Skin", "Insulin", "BMI"]
    result = ""

    if request.method == 'POST':
        try:
            data = [float(request.form[f]) for f in fields]
            pred = predict(diabetes_model, data)
            result = "No Diabetes ✅" if pred == 0 else "Diabetes Detected ⚠️"
        except:
            result = "Invalid Input ❌"

    return render_template("index.html", disease="Diabetes", fields=fields, result=result)

# ================= HEART =================
@app.route('/heart', methods=['GET', 'POST'])
def heart():
    fields = ["Age", "Sex", "CP", "Chol", "Thalach", "Oldpeak"]
    result = ""

    if request.method == 'POST':
        try:
            data = [float(request.form[f]) for f in fields]
            pred = predict(heart_model, data)
            result = "Healthy ❤️" if pred == 0 else "Heart Disease ⚠️"
        except:
            result = "Invalid Input ❌"

    return render_template("index.html", disease="Heart Disease", fields=fields, result=result)

# ================= PARKINSON =================
@app.route('/parkinson', methods=['GET', 'POST'])
def parkinson():
    fields = ["Fo", "Fhi", "Flo", "Jitter", "Shimmer", "HNR"]
    result = ""

    if request.method == 'POST':
        try:
            data = [float(request.form[f]) for f in fields]
            pred = predict(parkinson_model, data)
            result = "Normal 🧠" if pred == 0 else "Parkinson Detected ⚠️"
        except:
            result = "Invalid Input ❌"

    return render_template("index.html", disease="Parkinson", fields=fields, result=result)

# ================= LIVER DISEASE =================
@app.route('/liver', methods=['GET', 'POST'])
def liver():
    fields = [
        "Age", "Total_Bilirubin", "Direct_Bilirubin",
        "Alkaline_Phosphotase", "Alamine_Aminotransferase",
        "Aspartate_Aminotransferase", "Total_Proteins",
        "Albumin", "Albumin_and_Globulin_Ratio"
    ]
    result = ""

    if request.method == 'POST':
        try:
            data = [float(request.form[f]) for f in fields]
            pred = predict(liver_model, data)
            result = "Healthy Liver 🟢" if pred == 0 else "Liver Disease Detected 🔴"
        except:
            result = "Invalid Input ❌"

    return render_template("index.html", disease="Liver Disease", fields=fields, result=result)

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)