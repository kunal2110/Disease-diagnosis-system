from sklearn.linear_model import LogisticRegression
from preprocess import load_data,preprocess_data,split_scale
from sklearn.metrics import accuracy_score
import pickle

df = load_data(r"C:\Users\deshm\OneDrive\Desktop\Medical diagnosis\Dataset\parkinsons.data")

df = df.drop("name",axis=1)
print(df.columns)

target = "status"
x,y = preprocess_data(df,target)

xtrain,xtest,ytrain,ytest,scaler = split_scale(x,y)

model = LogisticRegression()
model.fit(xtrain,ytrain)
yp=model.predict(xtest)

print(f"acc:{accuracy_score(ytest,yp)}")
tr = model.score(xtrain,ytrain)
test = model.score(xtest,ytest)
print(tr,test)

pickle.dump(model,open("models/parkinson_model.pkl","wb"))
pickle.dump(scaler,open("models/parkinson_scaler.pkl","wb"))