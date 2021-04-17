#!/usr/bin/env python
# coding: utf-8

# In[11]:


import random
import numpy
import statistics
from pandas import DataFrame, read_csv
import pandas
from decimal import Decimal
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import csv
from sklearn import preprocessing 

datalatih=pandas.read_csv("data_training.csv")     #baca data
def detect():
    datalatih.isnull()#detect mising value


# In[23]:


att1=datalatih['Att1']
x1=statistics.median(att1)
attlist1 = [x for x in att1 if str(x) != 'nan']#pembuatan list att1 masih ori
att7=datalatih['Class']
classlist = [x for x in att7 if str(x) != 'nan']#pembuatan list class


# In[3]:


#pembuatan list att2 dan 3 terus encode nyari median juga terus masukin kelist masih ori
att2=datalatih['Att2']
att3=datalatih['Att3']
a=[]
attlist2=[]
for x in range (len(att2)):
    if att2[x]=='Besar':
        a='3'
    if att2[x]=='Sedang':
        a='2'
    if att2[x]=='Kecil':
        a='1'
    attlist2.append(a)
del[a]
a=[]
attlist3=[]
for i in range (len(att3)):
    if att3[i]=='Biru':
        a='3'
    if att3[i]=='Merah':
        a='2'
    if att3[i]=='Hijau':
        a='1'
    attlist3.append(a)
attlist2 = [int(i) for i in attlist2]
attlist3 = [int(i) for i in attlist3]
x2=statistics.median(attlist2)
x3=statistics.median(attlist3)


# In[4]:


#pembuatan list att4 masih ori
att4=datalatih['Att4']
attlist4 = [x for x in att4 if str(x) != 'nan']


# In[5]:


#pembuatan list att5 masih ori
att5=datalatih['Att5']
attlist5 = [x for x in att5 if str(x) != 'nan']
x5=statistics.median(attlist5)


# In[6]:


#pembuatan list att6 masih ori
att6=datalatih['Att6']
attlist6 = [x for x in att6 if str(x) != 'nan']


# In[7]:


#bikin boxplot data masih ori
data_to_plot = [attlist1, attlist2, attlist3, attlist4,attlist5,attlist6]
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(data_to_plot)


# In[8]:


#nilai sclar
x = datalatih.iloc[:, 3:5].values 
min_max_scaler = preprocessing.MinMaxScaler(feature_range =(0, 1)) 
x_after_min_max_scaler = min_max_scaler.fit_transform(x) 
att4scal=[]
att5scal=[]
print(x_after_min_max_scaler)
i=0
for i in range(11):
    att4scal.append(x_after_min_max_scaler[i][0])
    att5scal.append(x_after_min_max_scaler[i][1])


# In[40]:


#bikin csv
import csv
masuk[0]=[]
masuk[0].append(attlist1[0])
masuk[0].append(attlist2[0])
masuk[0].append(attlist3[0])
masuk[0].append(att4scal[0])
masuk[0].append(att5scal[0])
masuk[0].append(attlist6[0])
masuk[0].append(classlist[0])

i=1
for i in range(10):
    i=i+1
    masuk[i]=[]
    masuk[i].append(attlist1[i])
    masuk[i].append(attlist2[i])
    masuk[i].append(attlist3[i])
    masuk[i].append(att4scal[i])
    masuk[i].append(att5scal[i])
    masuk[i].append(attlist6[i])
    masuk[i].append(classlist[i])
row_list = [
    ['Att1', 'Att2','Att3','Att4','Att5','Att6','Class'],
masuk[0],masuk[1],masuk[2],masuk[3],masuk[4],masuk[5],masuk[6],masuk[7],masuk[8],masuk[9],masuk[10]
]
with open('data_training1.csv', 'w', newline='') as file:
    writer = csv.writer(file, escapechar='/', quoting=csv.QUOTE_NONE)
    writer.writerows(row_list)


# In[19]:


#gambar corelasi
dataubah=pandas.read_csv("data_training1.csv")     #baca data
df=dataubah[0:10]
def correlation_matrix(df):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    labels=['','att1','att2','att3','att4','att5','att6']
    ax1.set_xticklabels(labels,fontsize=10)
    ax1.set_yticklabels(labels,fontsize=10)
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()

correlation_matrix(df)


# In[41]:


#split,knn,akurasi


#Load the necessary python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#open data set
df = pd.read_csv('data_training1.csv')
X = df.drop('Class',axis=1).values
y = df['Class'].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)
#import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

#Setup arrays to store training and test accuracies
neighbors = np.arange(1,6)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #Fit the model
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test) 
#Generate plot
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='validating accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)
#import confusion_matrix
from sklearn.metrics import confusion_matrix
#let us get the predictions using the classifier we had fit above
y_pred = knn.predict(X_test)
confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
#import classification_report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
y_pred_proba = knn.predict_proba(X_test)[:,1]


# In[ ]:





# In[ ]:




