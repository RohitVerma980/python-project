#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Load Breast Cancer Data

# In[2]:


from sklearn.datasets import load_breast_cancer
cancer_data = load_breast_cancer()


# # Create DataFrame

# In[3]:


cancer = pd.DataFrame(data=cancer_data.data, columns=cancer_data.feature_names)


# In[4]:


cancer['target'] = cancer_data.target


# In[5]:


cancer


# In[6]:


cancer.isnull().sum()


# In[7]:


cancer.describe()


# # Plot Pairplot 

# In[8]:


sns.pairplot(cancer, hue='target', vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity'], diag_kind='kde')
plt.show()


# # Plot Countplot

# In[9]:


ax = sns.countplot(data=cancer, x='target')
ax.bar_label(ax.containers[0])


# # Plot Countplot on Mean radius

# In[10]:


cancer['mean radius bin'] = pd.cut(cancer['mean radius'], bins=10)
plt.figure(figsize=(20,8))
an = sns.countplot(data=cancer, x='mean radius')
an.bar_label(an.containers[0])
plt.xticks(rotation=45)
plt.legend()
plt.show()


# # Find Correlation

# In[11]:


cancer = cancer.drop('mean radius bin', axis=1)


# In[12]:


correlation_matrix = cancer.corr()


# # Plotting Correlation heatmap

# In[13]:


plt.figure(figsize=(20,15))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.show()


# # plotting correlation barplot

# In[14]:


cancer_2 = cancer.drop(['target'], axis=1)


# In[15]:


plt.figure(figsize=(16,9))
plt.bar(cancer_2.corrwith(cancer.target).index, cancer_2.corrwith(cancer.target).values)
plt.show()


# # split the data into train_test

# In[34]:


X = cancer_data.data  
y = cancer_data.target


# In[35]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)


# # Scale the data with standard scaler

# In[36]:


from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)


# # Logistic Regression

# In[37]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)


# In[39]:


y_pred = model.predict(x_test)


# In[41]:


model.score(x_test, y_pred)


# # Import acuracy score

# In[43]:


from sklearn.metrics import accuracy_score


# In[44]:


train_predictions = model.predict(x_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print(f'Training Accuracy: {train_accuracy}')


# In[45]:


test_predictions = model.predict(x_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f'Test Accuracy: {test_accuracy}')


# # Support vector machine

# In[46]:


from sklearn.svm import SVC


# In[47]:


svm = SVC(kernel='rbf')
svm.fit(x_train, y_train)


# In[48]:


prediction = svm.predict(x_test)


# In[50]:


svm.score(x_test, prediction)


# In[51]:


train_predictions = svm.predict(x_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print(f'Training Accuracy: {train_accuracy}')


# In[52]:


test_predictions = svm.predict(x_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f'Test Accuracy: {test_accuracy}')


# # Decision Tree

# In[71]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
tree.fit(x_train, y_train)


# In[72]:


pre = tree.predict(x_test)


# In[73]:


tree.score(x_test, pre)


# In[74]:


train_predictions = tree.predict(x_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print(f'Training Accuracy: {train_accuracy}')


# In[75]:


test_predictions = tree.predict(x_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f'Test Accuracy: {test_accuracy}')


# # Random Forest

# In[59]:


from sklearn.ensemble import RandomForestClassifier
forest =  RandomForestClassifier(n_estimators=100)
forest.fit(x_train, y_train)


# In[61]:


forex = forest.predict(x_test)


# In[62]:


forest.score(x_test, forex)


# In[63]:


train_predictions = forest.predict(x_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print(f'Training Accuracy: {train_accuracy}')


# In[64]:


test_predictions = forest.predict(x_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f'Test Accuracy: {test_accuracy}')


# # naive_bayes

# In[65]:


from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)


# In[66]:


nb = nb_classifier.predict(x_test)


# In[68]:


nb_classifier.score(x_test, nb)


# In[69]:


train_predictions = nb_classifier.predict(x_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print(f'Training Accuracy: {train_accuracy}')


# In[70]:


test_predictions = forest.predict(x_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f'Test Accuracy: {test_accuracy}')


# # Conclusion

# In[ ]:


#As you can out of all the models we have used svm is giving us the most accurate data without any overfitting. so, we'll go with svm

