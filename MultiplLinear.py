import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
df = pd.read_csv("Country_Dataset.csv")
x= df.drop(['Country', 'gdpp'], axis=1)
y = df['gdpp']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

print('Train Score: ', regressor.score(x_train, y_train))
print('Test Score: ', regressor.score(x_test, y_test))
# x = sm.add_constant(x)
#
# # Fit the multiple linear regression model
# model = sm.OLS(y, x).fit()
#
# # Print the model summary
# print(model.summary())