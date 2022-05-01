#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df=pd.read_csv("diabetes.csv")


# In[7]:


df


# In[14]:


plt.scatter(df.BMI,df.Outcome,marker='+',color='red')


# In[15]:


from sklearn.model_selection import train_test_split


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(df[['BMI']],df.Outcome,test_size=0.6)


# In[38]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[39]:


model.fit(X_train, y_train)


# In[40]:


model.score(X_test,y_test)


# In[41]:


model.predict([[39.8]])


# In[42]:


model.predict([[27.1]])


# In[ ]:




