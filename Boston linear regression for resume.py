import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')



"Generates a dataframe object"
print("Enter a csv file name:")
filename = input()
print("Enter the field you wish to find a fit for:")
fit = input()
df=[]
if filename!="" and match != "":
    df = pd.read_csv(filename)
else:
    fit = "Price"
    from sklearn.datasets import load_boston
    boston = load_boston()
    df = pd.DataFrame(boston["data"],columns = boston["feature_names"])
    price = pd.DataFrame(boston["target"], columns = [fit])
    df = pd.concat([df, price], axis=1)


# In[ ]:





# In[24]:


from sklearn.model_selection import train_test_split as tts
X = df.loc[:, df.columns != fit]
Y = df[fit]
x_train, x_test, y_train, y_test = tts(X,Y, test_size=0.3)


# In[25]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)


# In[26]:


coeffs = pd.DataFrame(lm.coef_.transpose(),X.columns,columns =['Coeff'])


# In[27]:


predictions = lm.predict(x_test)
distplot = sns.distplot((y_test-predictions),bins=30, axlabel = "Test-Prediction");
#img = mp.scatter(y_test, predictions)


# In[28]:


MaxCoeffID = coeffs['Coeff'].abs().idxmax()
plt = sns.jointplot(x=fit, y=MaxCoeffID, data = df, kind = "reg")


# In[29]:


from sklearn import metrics

print('Mean Absolute error', metrics.mean_absolute_error(y_test,predictions))

print('Mean Squared error', metrics.mean_squared_error(y_test, predictions))

print('Root Mean Squared error', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




