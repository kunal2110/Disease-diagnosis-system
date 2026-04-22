import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from preprocess import load_data,preprocess_data,split_scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression

df = load_data(r"C:\Users\deshm\OneDrive\Desktop\Medical diagnosis\Dataset\indian_liver_patient.csv")
print(df.columns)

target = "Dataset"
x,y = preprocess_data(df,target)
print(y.unique())
y = y.map({1:0, 2:1})

xtrain,xtest,ytrain,ytest,scaler = split_scale(x,y)
"""model = RandomForestClassifier( n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)
model.fit(xtrain,ytrain)
yp=model.predict(xtest)
tr_score = model.score(xtrain,ytrain)
test_score = model.score(xtest,ytest)
print(tr_score,test_score)

print(f"classification report:{classification_report(ytest,yp)}")
print(f"accuracy:{accuracy_score(ytest,yp)}")"""
model = LogisticRegression()
model.fit(xtrain,ytrain)
yp=model.predict(xtest)
print(accuracy_score(ytest,yp))
training_acc=model.score(xtrain,ytrain)
test_acc = model.score(xtest,ytest)
print(training_acc,test_acc)


pickle.dump(model,open("models/Liver_model.pkl","wb"))
pickle.dump(scaler,open("models/Liver_scaler.pkl","wb"))


print("model is saved successfully")