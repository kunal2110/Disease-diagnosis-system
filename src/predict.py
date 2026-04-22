import pickle
import numpy as np
import os
import pandas as pd

# 🔹 Base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")


# 🔹 LOAD MODEL (MUST BE ABOVE predict())
def load_model(model_name):
    try:
        model_path = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
        scaler_path = os.path.join(MODEL_DIR, f"{model_name}_scaler.pkl")

        model = pickle.load(open(model_path, "rb"))
        scaler = pickle.load(open(scaler_path, "rb"))

        return model, scaler

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


# 🔹 PREDICT FUNCTION
import pandas as pd

def predict(model_name, input_data):
    model, scaler = load_model(model_name)

    if model is None or scaler is None:
        return "Model not found"

    try:
        input_array = np.array(input_data).reshape(1, -1)

        # 🔹 Try to load columns (only for some models)
        columns_path = os.path.join(MODEL_DIR, f"{model_name}_columns.pkl")

        if os.path.exists(columns_path):
            # Models with get_dummies
            columns = pickle.load(open(columns_path, "rb"))

            input_df = pd.DataFrame([input_data])
            input_df = pd.get_dummies(input_df)
            input_df = input_df.reindex(columns=columns, fill_value=0)

            input_scaled = scaler.transform(input_df)

        else:
            # Models without get_dummies (diabetes, parkinsons)
            input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)

        return int(prediction[0])

    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error in prediction"

# 🔹 TEST BLOCK
if __name__ == "__main__":
    sample = [50, 1, 2, 120, 200, 0, 1, 150, 0, 2.3, 1, 0, 2]

    result = predict("heart", sample)
    print("Prediction:", result)