import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("orthopedic_features.csv")
le=LabelEncoder()
df['class']=le.fit_transform(df['class'])
# df["class"]=df['class'].map({'Normal':1,'Abnormal':0})


x=df.iloc[:,0:6].values
y=df.iloc[:,6].values
df2=pd.DataFrame(x)
print(df2.to_string())


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
print("x_train b4 sealing")
df3=pd.DataFrame(x_train)
print(df3.to_string())

from sklearn.preprocessing import StandardScaler
st_x=StandardScaler()
x_train=st_x.fit_transform(x_train)
x_test=st_x.transform(x_test)

print("x_train after scalind...")
df4=pd.DataFrame(x_train)
print(df4.to_string())


from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print(y_pred)
print("Predicition comparision")
ddf=pd.DataFrame({"Y_test":y_test,"Y-pred":y_pred})
print(ddf.to_string())

accuracy=accuracy_score(y_test,y_pred)
print('Accuracy:%.2f'%(accuracy*100))
