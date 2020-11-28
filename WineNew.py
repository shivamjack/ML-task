#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


Wine=pd.read_csv('winenew.csv')


# In[3]:


Wine.head()


# In[4]:


Wine.tail()


# In[5]:


Wine['quality'].value_counts()


# In[6]:


Wine.info()


# In[7]:


X=Wine.iloc[:,:-1]
X.head()


# In[9]:


y = Wine.iloc[:,-1:]
y.head()


# In[10]:


#data Normalization
from sklearn import preprocessing
X =preprocessing.StandardScaler().fit_transform(X)


# In[11]:


X[0:4]


# In[12]:


#Train Test split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)


# In[13]:


y_test.shape


# In[14]:


#Training and Predicting
from sklearn.neighbors import KNeighborsClassifier
knnmodel = KNeighborsClassifier(n_neighbors = 3)
knnmodel.fit(X_train,y_train)
ypredict = knnmodel.predict(X_test)


# In[15]:


#Accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,ypredict)
acc


# In[16]:


#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,ypredict)
cm


# In[22]:


cm1 = pd.DataFrame(data=cm,index=['good','bad'],columns=['good','bad'])
cm1


# In[23]:


#output visualization
prediction_output =pd.DataFrame(data=[y_test.values,ypredict],index=['y_test','ypredict'])


# In[24]:


prediction_output.transpose()


# In[21]:


#Finding the values of k
ks =21
mean_acc = np.zeros((ks-1))

for n in range(1,ks):
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n-1]=accuracy_score(y_test,yhat)
    


# In[25]:


print(mean_acc)


# In[26]:


print("The best accuracy was with",mean_acc.max(),"with k=",mean_acc.argmax()+1)


# In[23]:


plt.plot(range(1,ks),mean_acc,'g')
plt.ylabel('Accuracy')
plt.xlabel('K')
plt.tight_layout()
plt.show()


# In[ ]:




