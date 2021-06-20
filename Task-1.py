#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation 
# # Data Science and Business analytics internship
# #By Janvi Lala
# ##  Task 1 - Prediction using Supervised ML
 

# ##  Predicting the percentage of an student based on number of study hours

# #### Step1 : Importing the Libraries

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


url = "http://bit.ly/w-data"
df = pd.read_csv(url)
print("Data imported successfully")
df.head()


# In[4]:


df.info()


# ###  Visualizing the Data set

# In[5]:


df.plot(x='Hours',y='Scores', style = 'o')
plt.title('Hours Vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# ###  Preparing the Data

# In[6]:


X = df.iloc[:,:-1].values
y = df.iloc[:,1].values


# ### Splitting data into training and testing sets

# from sklearn.model_selection import train_test_split

# In[8]:


X_train,X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2, random_state = 0)


# ### Training the Data set

# In[9]:


from sklearn.linear_model import LinearRegression
l_reg = LinearRegression()
l_reg.fit(X_train,y_train)
print("Training Completed")


# In[10]:


print("Intercept = ",l_reg.intercept_)
print("Coefficient = ",l_reg.coef_)


# ### Plotting the Regression Line

# In[11]:



line = l_reg.coef_*X+l_reg.intercept_

#plotting for the test data
plt.scatter(X,y)
plt.plot(X,line)
plt.show()


# ### Making Predictions

# In[12]:


print("Hours = \n",X_test)
y_pred = l_reg.predict(X_test)
print("Predicted Scores = \n",y_pred)


# ### Comparing actual and predicted scores

# In[13]:


df_comp = pd.DataFrame({'Actual':y_test, 'Predicted': y_pred})
df_comp


# #### As per given task, we have to predict score of student if he/she studied for 9.25 hrs/day

# In[28]:


hours = 9.25
pred = l_reg.predict([[hours]])
print("Number of hours = {}".format(hours))
print("Predicted Score = {}".format(pred[0]))


# ### Evaluating using mean absolute error

# In[35]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[34]:


print("Mean Squared Error: ", metrics.mean_squared_error(y_test,y_pred))


# In[31]:


print("Mean Absolute Error: ", metrics.mean_absolute_error(y_test,y_pred))


# In[32]:


print("r2_score: ", r2_score(y_test,y_pred))

