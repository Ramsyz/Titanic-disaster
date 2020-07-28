#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[31]:


from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 

from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import accuracy_score

import sklearn.neighbors as neighbors
from sklearn.neighbors import KNeighborsClassifier


# In[19]:


df = pd.read_csv('train.csv')


# In[20]:


df.head()


# In[21]:



df.shape


# In[22]:


# Percentage of missing values in dataframe
df.isnull().sum().sort_values(ascending=False)
(df.isnull().sum()/df.isnull().sum().count()).sort_values(ascending=False)


# In[23]:


# numeric and cateogrical variables
df_num = df[['Age','SibSp','Parch', 'Fare']]
df_cat = df[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]

# # correlation
print(df_num.corr())
sns.heatmap(df_num.corr())


# In[24]:


# survival rate of Age Age,SibSp,Parch,Fare
pd.pivot_table(df, index='Survived', values=['Age','SibSp','Parch', 'Fare'])


# In[32]:


for i in df_cat.columns:
    sns.barplot(df_cat[i].value_counts().index, df_cat[i].value_counts()).set_title(i)
    plt.show()


# In[39]:


# Comparing survived by each category
pd.pivot_table(df, index = 'Survived', columns= 'Sex', values= 'Ticket', aggfunc= 'count')
pd.pivot_table(df, index = 'Survived', columns= 'Pclass', values= 'Ticket', aggfunc= 'count')
pd.pivot_table(df, index = 'Survived', columns= 'Embarked', values= 'Ticket', aggfunc= 'count')


# # Data  Clean and Preprocessing

# In[242]:


# Split the Name column from suffix Names
df = pd.read_csv('/kaggle/input/titanic/train.csv')

suffix_Name= df['Name'].str.replace(r'[^\w\s]+','')
suffix_Name= df['Name'].str.split(expand= True)

# Combine suffix Name to data frame
df = pd.concat([df,suffix_Name],axis=1)

df.head()


# In[243]:


# # Del observations without Embarked and getting index where it is null
# df.drop(df[pd.isnull(df['Embarked'])].index, inplace=True) 
# df[pd.isnull(df['Embarked'])]


# In[244]:


# Drop Cabin
#df.drop('Cabin', axis=1, inplace=True)
df.drop(columns=['PassengerId','Cabin'], axis=1, inplace=True)


# In[245]:


# (Snippet) Replacing missing values using the mean value of the columns that contain the missing values
age_imputer = SimpleImputer(strategy='median')
df['Age'] = age_imputer.fit_transform(df.loc[:,['Age']])
df['Age'].head()


# In[246]:


df['Embarked']=df['Embarked'].fillna(method='bfill')
df['Embarked'].count()


# In[247]:


# Rename column suffix name  as title
df.rename(columns={1:'Title'}, inplace=True)


# In[248]:


# remove leaky columns
#del df['Name']
#del df[0]
df.drop(columns=['Name',13,12,11,10,9,8,7,6,5,4,3,2,0], axis=1, inplace=True)


# In[249]:


df.head()


# In[250]:


# Label encode the variables
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder 

d = defaultdict(LabelEncoder)
#With this, you now retain all columns LabelEncoder as dictionary.

training_data = df.apply(lambda x: d[x.name].fit_transform(x))
training_data


# In[251]:


# test train split
y = training_data.Survived
X = training_data.drop(columns='Survived')

# Spitting data in two here is our model is learining patterns that will extend to novel patterns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


# In[252]:


training_data.head()


# # XGBoost

# In[262]:



# Convert to XGBoost format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

#num_class = y.nunique()


# In[263]:


# train model
param = {'objective':'multi:softmax', 'num_class': 6, 'max_depth': 10}
watchlist = [(dtrain, 'train'), (dtest, 'test')]

num_round = 10
bst = xgb.train(param, dtrain, num_round,watchlist)


# In[264]:


# get prediction
pred = bst.predict(dtest)
error_rate = np.sum(pred != y_test) / y_test.shape[0]
print('Test error using softmax = {}'.format(error_rate))


# In[265]:


# do the same thing again, but output probabilities
param['objective'] = 'multi:softprob'
num_round = 15
bst_prob = xgb.train(param, dtrain, num_round, watchlist)


# In[266]:


# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
pred_prob = bst_prob.predict(dtest).reshape(y_test.shape[0],6)
pred_label = np.argmax(pred_prob, axis=1)
error_rate = np.sum(pred_label != y_test) / y_test.shape[0]
print('Test error using softprob = {}'.format(error_rate))


# In[267]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, pred_label)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# # Cross validation (KFlod)
# 
# accuracy of a linear 'kernel'(we cahnge to 'rbf','poly','sigmoid') support vector machine on dataset by splitting the data, fitting a model 
# and computing the score 10 consecutive times

# In[268]:


from sklearn import svm
from sklearn.model_selection import cross_val_score

X_train.shape, y_train.shape
X_test.shape, y_test.shape

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
#clf.score(X_test, y_test)
scores = cross_val_score(clf, X, y, cv=10)
scores


# CV iteration is the score method estimator,possible to change this by using the scoring parameter

# In[269]:


from sklearn import metrics
scores = cross_val_score(
clf, X, y, cv=10, scoring='f1_macro')
scores


# # KNN

# In[270]:


import sklearn.neighbors as neighbors
from sklearn.neighbors import KNeighborsClassifier
# neighbors.BallTree

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)


# In[272]:



pred_label = neigh.predict(X_test)
error_rate = np.sum(pred_label != y_test) / y_test.shape[0]

print(error_rate)


# In[273]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, pred_label)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:




