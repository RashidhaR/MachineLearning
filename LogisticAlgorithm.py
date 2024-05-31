import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


df = pd.read_csv("orthopedic_features.csv")
print(df.to_string())

le=LabelEncoder()
df['class']=le.fit_transform(df['class'])

X=df.drop('class',axis=1)
y = df['class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

df2 = pd.DataFrame(X_test)
print(df2.to_string())

print(y_pred)

df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df2.to_string())

from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print("Accuracy:", accuracy_score(y_test, y_pred))
