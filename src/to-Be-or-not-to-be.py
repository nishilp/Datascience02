
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns


# In[2]:


shakespeare_data_url = "https://raw.githubusercontent.com/nishilp/Datascience02/master/data/Shakespeare_data.csv"


# In[3]:


# Loading the data into pandas dataframe

full_data = pd.read_csv(shakespeare_data_url)


# In[4]:


# Verifying the pandas dataframe

full_data.head(5)


# In[5]:


# Checking for duplicate rows

sum(full_data.duplicated())

# No duplicate found


# In[6]:



# Check total number of rows and columns. This frame has 5320 rows and 14 columns

full_data.shape

# This dataset has 111396 rows and 6 columns


# In[7]:


# Check rows with missing attribute values

full_data.isnull().sum(axis=0)

# Interesting revelation here is that attribute "ActSceneLine" has 6243 missing values


# In[8]:


# Almost all "PlayerLine" values are unique, so its better to drop it from our dataframe to simplify classification
full_data = full_data.drop(columns=['PlayerLine'])
full_data = full_data.drop(columns=['Dataline'])
full_data = full_data.drop(columns=['ActSceneLine'])


# In[9]:


full_data.head(5)


# In[10]:


# We need to convert categorical features to numerical ones
# Using LabelEncoder on our target feature

from sklearn.preprocessing import LabelEncoder

number = LabelEncoder()
full_data['Player'] = number.fit_transform(full_data['Player'].astype('str'))
# full_data['PlayerLine'] = number.fit_transform(full_data['PlayerLine'].astype('str'))


# In[11]:


# Label encoded target "Player"

full_data.head(5)


# In[12]:


full_data.dtypes


# In[13]:


# Onehot encoding on feature "Play"

full_data = pd.get_dummies(full_data, drop_first=True)


# In[14]:


# Verifying the data after onehot encoding of categorical features (in this case "Play")

full_data.head(5)


# In[15]:


# Dropping rows with null values 

full_data = full_data.dropna()


# In[16]:


# Splitting the data into training and testing sets using sklearn module "train_test_split"

X = full_data.drop(columns=['Player'])
y = full_data['Player']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[17]:


# Classification Model : Logistic Regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))


# In[18]:


# Classification Model : Decision Trees

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier().fit(X_train, y_train)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


# In[19]:


# Classification Model : K-nearest neighbors

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))


# In[20]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'
     .format(lda.score(X_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'
     .format(lda.score(X_test, y_test)))


# In[21]:


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
     .format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
     .format(gnb.score(X_test, y_test)))

