#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


# In[2]:


data = pd.read_csv("C:\\Users\\ROHIT\\Desktop\\data\\titanic.csv")
data


# In[3]:


data.isnull().sum()


# In[4]:


data.drop('Embarked', axis=1,inplace=True)
data.drop('PassengerId', axis=1,inplace=True)
data.drop('Name', axis=1,inplace=True)
data.drop('SibSp', axis=1,inplace=True)
data.drop('Parch', axis=1,inplace=True)
data.drop('Ticket', axis=1,inplace=True)
data.drop('Cabin', axis=1,inplace=True)


# In[5]:


data


# In[6]:


data['Age'].fillna(data['Age'].median(), inplace=True)


# In[7]:


Q1 = data['Age'].quantile(0.25)
Q3 = data['Age'].quantile(0.75)
IQR = Q3 - Q1

# Define the bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = data[(data['Age'] < lower_bound) | (data['Age'] > upper_bound)]
print(f"Outliers:\n{outliers}")


# In[9]:


df = data[(data['Age'] >= lower_bound) & (data['Age'] <= upper_bound)]


# In[10]:


df


# In[13]:


dummy = pd.get_dummies(df['Sex'])
dummy = dummy.applymap(lambda x: int(x) if isinstance(x, bool) else x)
dummy


# In[14]:


df = pd.concat([df, dummy], axis=1)
df


# In[23]:


#data.drop('Sex', axis=1,inplace=True)
df.drop(['Sex'], axis=1,inplace=True)
df


# In[26]:


x = df.drop('Survived', axis=1)
y = df['Survived']


# In[27]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=0)


# In[29]:


model = DecisionTreeClassifier(criterion='gini')
model.fit(x_train,y_train)


# In[33]:


y_pred =model.predict([[1, 35.0, 100, 0]])


# In[31]:


x_test


# In[36]:


model.score(x_train, y_train)


# In[ ]:




