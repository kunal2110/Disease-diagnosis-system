import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

from preprocess import load_data,preprocess_data,split_scale
#3.load_data
df = load_data(r"C:\Users\deshm\OneDrive\Desktop\Medical diagnosis\Dataset\diabetes_prediction_dataset.csv")
print(df.columns)
df['gender'] = df['gender'].map({'Male':1, 'Female':0})
df['smoking_history'] = df['smoking_history'].astype('category').cat.codes
#4.separate target col
target = "diabetes"
x,y = preprocess_data(df,target)
#5.split and scale
xtrain,xtest,ytrain,ytest,scaler = split_scale(x,y)
#6.model building
model = RandomForestClassifier()
model.fit(xtrain,ytrain)
yp = model.predict(xtest)
#7.model evaluation
print(f"accuracy_score:{accuracy_score(ytest,yp)}")
tr_score = model.score(xtrain,ytrain)
test_score = model.score(xtest,ytest)
print(tr_score,test_score)
#8. Save model
pickle.dump(model, open("models/diabetes_model.pkl", "wb"))
pickle.dump(scaler, open("models/diabetes_scaler.pkl", "wb"))

print("Model saved successfully!")
