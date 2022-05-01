#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("diabetes.csv")


# In[3]:


df


# In[4]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


inputs=df.drop('Outcome',axis='columns')
target=df['Outcome']


# In[54]:


plt.xlabel('BMI')
plt.ylabel('Glucose')
plt.scatter(df['BMI'], df['Glucose'],color="green",marker='+')
plt.scatter(df['Glucose'], df['BMI'],color="blue",marker='.')


# In[55]:


from sklearn.model_selection import train_test_split


# In[56]:


x=df.drop(['Outcome','Age','Pregnancies','BloodPressure','SkinThickness','Insulin','DiabetesPedigreeFunction'],axis='columns')
y=df['Outcome']


# In[57]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[58]:


len(x_train)


# In[59]:


len(x_test)


# In[60]:


from sklearn.svm import SVC
model = SVC()


# In[61]:


model.fit(x_train,y_train)


# In[62]:


model.score(x_test,y_test)


# In[63]:


model.predict([[73,23]])


# In[65]:


model.predict([[167,37]])


# In[ ]:




