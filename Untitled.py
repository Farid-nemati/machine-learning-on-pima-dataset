#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df= pd.read_csv("diabetes.csv")


# In[3]:


df


# In[4]:


inputs=df.drop('Outcome',axis='columns')
target=df['Outcome']


# In[5]:


inputs


# In[6]:


target


# In[7]:


from sklearn import tree


# In[8]:


model = tree.DecisionTreeClassifier()


# In[9]:


model.fit(inputs,target)


# In[10]:


model.score(inputs,target)


# In[13]:


model.predict([[6,148,72,35,0,33.6,0.627,50]])


# In[15]:


model.predict([[1,89,66,23,94,28.1,0.167,21]])

