import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sns
import matplotlib.pyplot as plt


'''
    This project does a linear fit for a csv file of your choice using macine learning
    filename default: Boston housing
    fit defult: Price
'''
print("Enter a csv file name:")
filename = str(input()) #try using: Ecommerce Customers
print("Enter the field you wish to find a fit for:")
fit = str(input())#try using: Yearly Amount Spent
df=[]
if filename!="" and fit != "":
    print("Filename: ", filename, " fit: ", fit)
    df = pd.read_csv(filename)
else:
    #default: no input was given, setting boston dataset
    fit = "Price"
    from sklearn.datasets import load_boston
    boston = load_boston()
    df = pd.DataFrame(boston["data"],columns = boston["feature_names"])
    price = pd.DataFrame(boston["target"], columns = [fit])
    df = pd.concat([df, price], axis=1)





#splitting the data into training data and testing data
from sklearn.model_selection import train_test_split as tts
X = df.loc[:, (df.columns != fit)]._get_numeric_data()
Y = df[fit]
x_train, x_test, y_train, y_test = tts(X,Y, test_size=0.3)


#training the data on the training set
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)

#generating predictions using the test data
predictions = lm.predict(x_test)

#plotting the distribution of mistakes, the closer this is to gausian the better
#if the didtribution is non gausian, linear fit probably isnt the best option
distplot = sns.distplot((y_test-predictions),bins=30, axlabel = "Test-Prediction");

#plotting the coefficiant with the largest impact
coeffs = pd.DataFrame(lm.coef_.transpose(),X.columns,columns =['Coeff'])
MaxCoeffID = coeffs['Coeff'].abs().idxmax()
plt = sns.jointplot(x=fit, y=MaxCoeffID, data = df, kind = "reg")


#printing metrics
from sklearn import metrics
print('Mean Absolute error', metrics.mean_absolute_error(y_test,predictions))
print('Mean Squared error', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared error', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
