import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#load data
def load_data(path):
    df = pd.read_csv(path)
    return df

#preprocess dataset
def preprocess_data(df,target_column):
    # split data in x and y
    x = df.drop(target_column,axis = 1)
    y = df[target_column]
    #convert categorical data into numeric data
    x = pd.get_dummies(x)
    x = x.fillna(x.median())
    return x,y
# split and scale data 
def split_scale(x,y):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.25,random_state=42)
    
    scaler = StandardScaler()
    xtrain=scaler.fit_transform(xtrain)
    xtest=scaler.transform(xtest)
    return xtrain,xtest,ytrain,ytest,scaler
