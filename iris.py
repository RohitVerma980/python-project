#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble  import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder


# In[3]:


iris = datasets.load_iris()


# In[4]:


iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df


# In[5]:


iris_df['species'] = iris.target


iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})


# In[12]:


iris_df.head()


# In[9]:


le = LabelEncoder()
iris_df['species'] = le.fit_transform(iris.target_names[iris.target])


# In[11]:


iris_df.tail()


# In[14]:


x = iris_df.drop('species', axis=1)
y = iris_df['species']


# In[15]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=0)


# In[20]:


model = RandomForestClassifier(n_estimators=20)
model.fit(x_train,y_train)


# In[21]:


model.predict([[6, 1, 2, 1]])


# In[22]:


model.score(x_test, y_test)


# In[ ]:




