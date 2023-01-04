#!/usr/bin/env python
# coding: utf-8

# In[127]:


import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

import os
for dirname, _, filenames in os.walk('./input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[6]:


desktopData= pd.read_csv("./input/Dyt-desktop.csv", index_col=0, na_values=['(NA)'])
tabletData = pd.read_csv("./input/Dyt-tablet.csv", index_col=0, na_values=['(NA)'])


# In[9]:


def SeparateColumns(dataSetName):
    columns = defaultdict(list)
    with open(dataSetName, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        headers = next(reader)
        column_nums = range(len(headers)) 
        for row in reader:
            for i in column_nums:
            
                columns[headers[i]].append(row[i])
    return dict(columns)


# In[10]:


def cleanData(data) :
    for col in data.columns.values:
        data[col] = data[col].astype('string')
    #----------
    for col in data.columns.values:
        data[col] = data[col].astype('float',errors = 'ignore')
    #-----------
    data['Gender']=data.Gender.map({'Male': 1, 'Female': 2})
    data['Dyslexia']=data.Dyslexia.map({'No': 0, 'Yes': 1})
    data['Nativelang']=data.Nativelang.map({'No': 0, 'Yes': 1})
    data['Otherlang']=data.Otherlang.map({'No': 0, 'Yes': 1})


# In[12]:


columns = SeparateColumns('./input/Dyt-desktop.csv')
desktopData=pd.DataFrame.from_dict(columns)

desktopData


# In[13]:


cleanData(desktopData)

desktopData.head()


# In[14]:


columns = SeparateColumns('./input/Dyt-tablet.csv')
tabletData=pd.DataFrame.from_dict(columns)
tabletData.replace(["NULL"], np.nan, inplace = True)

tabletData


# In[15]:


cleanData(tabletData)

tabletData.head()


# In[16]:


stateOfNUll= tabletData.isnull().any()
i = 0
for state in stateOfNUll : 
    if(state):  
        tabletData[stateOfNUll.index[i]].fillna(round(tabletData[stateOfNUll.index[i]].mean() , 4), inplace=True)
    i = i + 1    

tabletData


# In[17]:


cols_with_missing = [col for col in tabletData.columns if tabletData[col].isnull().any()]

# Drop columns desktop data
reduced_desktopData = desktopData.drop(cols_with_missing, axis=1)

# Drop columns tablet data
reduced_tabletData = tabletData.drop(cols_with_missing, axis=1)


# In[18]:


commonalityColumns = ['Gender','Nativelang','Otherlang','Age' , 'Dyslexia']
for i in  range(30):
    if((i>=0 and i<12) or (i>=13 and i<17) or i==21 or i==22 or i==29):
        commonalityColumns.append('Clicks'+str(i+1))
        commonalityColumns.append('Hits'+str(i+1))
        commonalityColumns.append('Misses'+str(i+1))
        commonalityColumns.append('Score'+str(i+1))
        commonalityColumns.append('Accuracy'+str(i+1))
        commonalityColumns.append('Missrate'+str(i+1))
    
reduced_desktopData=reduced_desktopData.loc[:,commonalityColumns]
reduced_tabletData=reduced_tabletData.loc[:,commonalityColumns]


# In[19]:


y=reduced_desktopData['Dyslexia']
X=reduced_desktopData.loc[:, reduced_desktopData.columns != 'Dyslexia']


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[21]:


#----RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train , y_train)
y_pred = rfc.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[22]:


yTest=reduced_tabletData['Dyslexia']
XTest=reduced_tabletData.loc[:, reduced_tabletData.columns != 'Dyslexia']


# In[23]:


rfc2 = RandomForestClassifier()
rfc2.fit(X_train , y_train)
y_pred = rfc2.predict(XTest)
print("Accuracy:",metrics.accuracy_score(yTest, y_pred))


# In[66]:


from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[159]:


knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y2_pred=knn.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y2_pred))


# In[68]:


yTest=reduced_tabletData['Dyslexia']
XTest=reduced_tabletData.loc[:, reduced_tabletData.columns != 'Dyslexia']


# In[69]:


knn2 = KNeighborsClassifier(n_neighbors=6)
knn2.fit(X_train , y_train)
y2_pred = knn2.predict(XTest)
print("Accuracy:",metrics.accuracy_score(yTest, y2_pred))


# In[70]:


from sklearn.ensemble import AdaBoostClassifier


# In[71]:


adb=AdaBoostClassifier()
adb.fit(X_train, y_train)
y_pred=adb.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[74]:


from sklearn.svm import SVC


# In[76]:


svc=SVC(kernel='poly', degree=1, gamma='auto')
svc.fit(X_train, y_train)
y_pred=svc.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[77]:


from sklearn.neural_network import MLPClassifier


# In[78]:


mlp=MLPClassifier()
mlp.fit(X_train, y_train)
y_pred=mlp.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[84]:


from sklearn.linear_model import LogisticRegression 
lr=LogisticRegression()
lr.fit(X_train, y_train)
y_pred=lr.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[85]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred=dtc.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[86]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred=lda.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[91]:


from sklearn.ensemble import VotingClassifier
# 1) naive bias = mnb
# 2) logistic regression =lr
# 3) random forest =rf
# 4) support vector machine = svm
evc=VotingClassifier(estimators=[('adb', adb),('lr',lr),('rfc',rfc),('svc',svc)],voting='hard')
evc.fit(X_train, y_train)
y_pred=evc.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[107]:


from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(8,8,8),activation='logistic',solver='adam',max_iter=500)
classifier.fit(X_train, y_train)



# Predicting the Test set results
y_pred = classifier.predict(X_test)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#Interpretation:
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#ACCURACY SCORE
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred)*100)


# In[112]:


from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(8,8,8),activation='tanh',solver='adam',max_iter=500)
classifier.fit(X_train, y_train)



# Predicting the Test set results
y_pred = classifier.predict(X_test)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#Interpretation:
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#ACCURACY SCORE
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred)*100)


# In[113]:


from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(8,8,8),activation='relu',solver='adam',max_iter=500)
classifier.fit(X_train, y_train)



# Predicting the Test set results
y_pred = classifier.predict(X_test)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#Interpretation:
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#ACCURACY SCORE
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred)*100)


# In[118]:


from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(8,8,8),activation='identity',solver='adam',max_iter=500)
classifier.fit(X_train, y_train)



# Predicting the Test set results
y_pred = classifier.predict(X_test)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#Interpretation:
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#ACCURACY SCORE
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred)*100)


# In[131]:


estimators = []

#Defining 5 Logistic Regression Models
model11 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic1', model11))
model12 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic2', model12))
model13 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic3', model13))
model14 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic4', model14))
model15 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic5', model15))

#Defining 5 Decision Tree Classifiers
model16 = DecisionTreeClassifier(max_depth = 3)
estimators.append(('cart1', model16))
model17 = DecisionTreeClassifier(max_depth = 4)
estimators.append(('cart2', model17))
model18 = DecisionTreeClassifier(max_depth = 5)
estimators.append(('cart3', model18))
model19 = DecisionTreeClassifier(max_depth = 2)
estimators.append(('cart4', model19))
model20 = DecisionTreeClassifier(max_depth = 3)
estimators.append(('cart5', model20))

#Defining 5 Support Vector Classifiers
model21 = SVC(kernel = 'linear')
estimators.append(('svm1', model21))
model22 = SVC(kernel = 'poly')
estimators.append(('svm2', model22))
model23 = SVC(kernel = 'rbf')
estimators.append(('svm3', model23))
model24 = SVC(kernel = 'rbf')
estimators.append(('svm4', model24))
model25 = SVC(kernel = 'linear')
estimators.append(('svm5', model25))

#Defining 5 K-NN classifiers
model26 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
estimators.append(('knn1', model26))
model27 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
estimators.append(('knn2', model27))
model28 = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2)
estimators.append(('knn3', model28))
model29 = KNeighborsClassifier(n_neighbors = 4, metric = 'minkowski', p = 1)
estimators.append(('knn4', model29))
model30 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 1)
estimators.append(('knn5', model30))

#Defining 5 Naive Bayes classifiers
model31 = GaussianNB()
estimators.append(('nbs1', model31))
model32 = GaussianNB()
estimators.append(('nbs2', model32))
model33 = GaussianNB()
estimators.append(('nbs3', model33))
model34 = GaussianNB()
estimators.append(('nbs4', model34))
model35 = GaussianNB()
estimators.append(('nbs5', model35))

# Defining the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

#Confision matrix
cm_HybridEnsembler = confusion_matrix(y_test, y_pred)

print(accuracy_score(y_test,y_pred)*100)


# In[166]:


estimators = []

#Defining 5 Logistic Regression Models
model11 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic1', model11))
model12 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic2', model12))
model13 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic3', model13))
model14 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic4', model14))
model15 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic5', model15))

#Defining 5 K-NN classifiers
model26 = KNeighborsClassifier(n_neighbors = 5)
estimators.append(('knn1', model26))
model27 = KNeighborsClassifier(n_neighbors = 5)
estimators.append(('knn2', model27))
model28 = KNeighborsClassifier(n_neighbors = 5)
estimators.append(('knn3', model28))
model29 = KNeighborsClassifier(n_neighbors = 5)
estimators.append(('knn4', model29))
model30 = KNeighborsClassifier(n_neighbors = 5)
estimators.append(('knn5', model30))

#Defining 5 Naive Bayes classifiers
model31 = GaussianNB()
estimators.append(('nbs1', model31))
model32 = GaussianNB()
estimators.append(('nbs2', model32))
model33 = GaussianNB()
estimators.append(('nbs3', model33))
model34 = GaussianNB()
estimators.append(('nbs4', model34))
model35 = GaussianNB()
estimators.append(('nbs5', model35))

# Defining the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

#Confision matrix
cm_HybridEnsembler = confusion_matrix(y_test, y_pred)

print(accuracy_score(y_test,y_pred)*100)


# In[140]:


estimators = []

#Defining 5 Logistic Regression Models
model11 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic1', model11))
model12 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic2', model12))
model13 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic3', model13))
model14 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic4', model14))
model15 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic5', model15))

#Defining 5 Decision Tree Classifiers
model16 = DecisionTreeClassifier(max_depth = 3)
estimators.append(('cart1', model16))
model17 = DecisionTreeClassifier(max_depth = 4)
estimators.append(('cart2', model17))
model18 = DecisionTreeClassifier(max_depth = 5)
estimators.append(('cart3', model18))
model19 = DecisionTreeClassifier(max_depth = 2)
estimators.append(('cart4', model19))
model20 = DecisionTreeClassifier(max_depth = 3)
estimators.append(('cart5', model20))


#Defining 5 Support Vector Classifiers
model21 = SVC(kernel = 'linear')
estimators.append(('svm1', model21))
model22 = SVC(kernel = 'poly')
estimators.append(('svm2', model22))
model23 = SVC(kernel = 'rbf')
estimators.append(('svm3', model23))
model24 = SVC(kernel = 'rbf')
estimators.append(('svm4', model24))
model25 = SVC(kernel = 'linear')
estimators.append(('svm5', model25))

#Defining 5 K-NN classifiers
model26 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
estimators.append(('knn1', model26))
model27 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
estimators.append(('knn2', model27))
model28 = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2)
estimators.append(('knn3', model28))
model29 = KNeighborsClassifier(n_neighbors = 4, metric = 'minkowski', p = 1)
estimators.append(('knn4', model29))
model30 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 1)
estimators.append(('knn5', model30))

#Defining 5 Naive Bayes classifiers
model31 = GaussianNB()
estimators.append(('nbs1', model31))
model32 = GaussianNB()
estimators.append(('nbs2', model32))
model33 = GaussianNB()
estimators.append(('nbs3', model33))
model34 = GaussianNB()
estimators.append(('nbs4', model34))
model35 = GaussianNB()
estimators.append(('nbs5', model35))

#Defining 5 AdaBoost Classifiers
model36 = AdaBoostClassifier()
estimators.append(('ada1', model36))
model37 = AdaBoostClassifier()
estimators.append(('ada2', model37))
model38 = AdaBoostClassifier()
estimators.append(('ada3', model38))
model39 = AdaBoostClassifier()
estimators.append(('ada4', model39))
model40 = AdaBoostClassifier()
estimators.append(('ada5', model40))

# Defining the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

#Confision matrix
cm_HybridEnsembler = confusion_matrix(y_test, y_pred)

print(accuracy_score(y_test,y_pred)*100)


# In[175]:


estimators = []

#Defining 5 Logistic Regression Models
model11 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic1', model11))
model12 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic2', model12))
model13 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic3', model13))
model14 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic4', model14))
model15 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic5', model15))


#Defining 5 Naive Bayes classifiers
model31 = GaussianNB()
estimators.append(('nbs1', model31))
model32 = GaussianNB()
estimators.append(('nbs2', model32))
model33 = GaussianNB()
estimators.append(('nbs3', model33))
model34 = GaussianNB()
estimators.append(('nbs4', model34))
model35 = GaussianNB()
estimators.append(('nbs5', model35))

# Defining the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

#Confision matrix
cm_HybridEnsembler = confusion_matrix(y_test, y_pred)

print(accuracy_score(y_test,y_pred)*100)


# In[180]:


estimators = []

#Defining 5 Logistic Regression Models
model11 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic1', model11))
model12 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic2', model12))
model13 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic3', model13))
model14 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic4', model14))
model15 = LogisticRegression(penalty = 'l2', random_state = 0)
estimators.append(('logistic5', model15))


#Defining 5 Naive Bayes classifiers
model31 = GaussianNB()
estimators.append(('nbs1', model31))
model32 = GaussianNB()
estimators.append(('nbs2', model32))
model33 = GaussianNB()
estimators.append(('nbs3', model33))
model34 = GaussianNB()
estimators.append(('nbs4', model34))
model35 = GaussianNB()
estimators.append(('nbs5', model35))

#Defining 5 AdaBoost Classifiers
model36 = AdaBoostClassifier()
estimators.append(('ada1', model36))
model37 = AdaBoostClassifier()
estimators.append(('ada2', model37))
model38 = AdaBoostClassifier()
estimators.append(('ada3', model38))
model39 = AdaBoostClassifier()
estimators.append(('ada4', model39))
model40 = AdaBoostClassifier()
estimators.append(('ada5', model40))

# Defining the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

#Confision matrix
cm_HybridEnsembler = confusion_matrix(y_test, y_pred)

print(accuracy_score(y_test,y_pred)*100)


# In[ ]:




