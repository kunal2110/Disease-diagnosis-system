import pickle 
from preprocess import load_data, split_scale, preprocess_data
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

df = load_data(r"C:\Users\deshm\OneDrive\Desktop\Medical diagnosis\Dataset\heart_disease_uci.csv")

print(df.columns)

# ✅ FIX 1: Drop useless column
df = df.drop("id", axis=1)

# ✅ FIX 2: Convert target to binary
df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

target = "num"

x, y = preprocess_data(df, target)

xtrain, xtest, ytrain, ytest, scaler = split_scale(x, y)

# ✅ Logistic Regression (best for this dataset)
model = LogisticRegression(max_iter=1000)

model.fit(xtrain, ytrain)

yp = model.predict(xtest)

print("Accuracy:", accuracy_score(ytest, yp))

training_score = model.score(xtrain, ytrain)
testing_score = model.score(xtest, ytest)

print("Train:", training_score)
print("Test:", testing_score)


pickle.dump(model,open("models/heart_model.pkl","wb"))
pickle.dump(scaler,open("models/heart_scaler.pkl","wb"))
 #Save column names (IMPORTANT for prediction)
pickle.dump(x.columns, open("models/heart_columns.pkl", "wb"))