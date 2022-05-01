#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df=pd.read_csv("diabetes.csv")


# In[4]:


df


# In[5]:


inputs=df.drop('Outcome',axis='columns')
target=df['Outcome']


# In[6]:


inputs


# In[7]:


target


# In[8]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)


# In[9]:


knn.fit(inputs,target)


# In[10]:


knn.score(inputs,target)


# In[12]:


knn = KNeighborsClassifier(n_neighbors=5)


# In[14]:


knn.score(inputs,target)


# In[16]:


knn.predict([[10,101,76,48,180,32.9,0.171,63]])


# In[17]:


knn.predict([[1,129,60,0,0,30.1,0.349,47]])


# In[ ]:




