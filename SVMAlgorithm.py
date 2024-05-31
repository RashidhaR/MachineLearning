import numpy as np
import matplotlib as mlt
import pandas as pd

df=pd.read_csv('orthopedic_features.csv')
df["class"]=df['class'].map({'Normal':1,'Abnormal':0})
x=df.drop('class',axis=1)
y = df['class']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
st_x=StandardScaler()
x_train=st_x.fit_transform(x_train)
x_test=st_x.transform(x_test)

from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
df2=pd.DataFrame({"Actual Y_Test":y_test,"PredictionData":y_pred})
print("prediction status")
print(df2.to_string())