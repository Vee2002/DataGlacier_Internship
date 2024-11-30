#!/usr/bin/env python
# coding: utf-8

# In[103]:


# Importing libraries I need

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Models
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,make_scorer,recall_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


# In[104]:


# Loading the dataset

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, 'Dataset', 'healthcare-dataset-stroke-data.csv')

data = pd.read_csv('Dataset/healthcare-dataset-stroke-data.csv')
data.head(5)


# In[105]:


data.info()


# From the above information, we can see that the column bmi has 4909 entries out of the 5110 which may mean that it has missing values.Let's confirm if it does

# In[106]:


data.isna().sum()


# There are 201 missing entries in the bmi.This is close to 4% of the values in that column that are missing.This is quite a small percentage of missing values so we can drop them

# In[107]:


data.dropna(subset=['bmi'],inplace=True)


# In[108]:


data.isna().sum()


# Now there are no missing values

# Checking for duplicates

# In[109]:


data.duplicated().sum()


# There are no duplicated values in the dataset

# In[110]:


# Checking for outliers

plt.figure(figsize=(10,8))
sns.boxplot(data)



# The data has no outliers

# In[111]:


# Dropping id column as it does not have much impact on the target variable

data.drop(columns=['id'],inplace=True)


# In[112]:


data


# In[113]:


data['gender'].value_counts()


# Gender other is only one out of all the gender values.This may have a very minute effect on our dataset hence let's drop it

# In[114]:


data.drop(data[data['gender'] == 'Other'].index,axis=0,inplace=True)


# In[115]:


# Confirming if the row has been dropped

data['gender'].value_counts()


# In[116]:


# Converting age column to int from float

data['age'] = data['age'].astype(int)


# In[117]:


# Checking for class imbalance

data['stroke'].value_counts(normalize=True)*100


# In this data, most people have no stroke as this category has carried 95% of the dataset.This may bring bias to our model so let's try correct it .

# In[118]:


data['stroke'].value_counts().plot(kind='bar',figsize=(10,8),title='Stroke Distribution')
plt.ylabel('Value Counts')
plt.xlabel('Stroke')


# ### ONE-HOT ENCODING

# The model can not take categorical data so we have to encode so that it can be input to the model

# In[119]:


data


# In[120]:


data['work_type'].value_counts()


# In[121]:


# Splitting into train and test
X = data.drop(columns=['stroke'])
y = data['stroke']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[122]:


data


# In[123]:


# Encoding using pd.get dummies

X_train_encoded = pd.get_dummies(X_train,columns=['gender','ever_married','work_type','Residence_type','smoking_status'],drop_first=True)
X_test_encoded = pd.get_dummies(X_test,columns=['gender','ever_married','work_type','Residence_type','smoking_status'],drop_first=True)


# In[124]:


# Converting true and false to 1 and 0 respectively
# pd.set_option('future.no_silent_downcasting',True)

X_train_encoded.replace({True:1,False:0},inplace=True)
X_test_encoded.replace({True:1,False:0},inplace=True)


# In[125]:


X_test_encoded


# Our data is now ready to fit into the model

# #### MODELLING

# In[126]:


# Correcting class imbalance
smote = SMOTE(random_state=42)

X_train_resampled,y_train_resampled = smote.fit_resample(X_train_encoded.values,y_train)

# Training random forest on the resampled data
lr = LogisticRegression(random_state=42,max_iter=1000)

# Fitting the model
lr.fit(X_train_resampled,y_train_resampled)

# Predicting the data
prediction = lr.predict(X_test_encoded.values)

# Evaluation metrics
print(f"Recall Score: {recall_score(y_test,prediction)}")


# Our model is performing quite well from the precision score results. The Logistic Regression is able to identify actual positive instances(Stroke instances) out of the total predictions

# #### DEPLYOMENT

# In[ ]:





# In[127]:


# Writing the model.py file

code = """ 
from Stroke_Prediction import X_train_resampled,X_test_encoded,y_train_resampled,y_test,lr,recall_score

# Using pickle to serialize/deserialize

import joblib

joblib.dump(X_train_resampled,'X_train_resampled.pkl')
joblib.dump(X_test_encoded,'X_test_encoded.pkl')
joblib.dump(y_train_resampled,'y_train_resampled.pkl')
joblib.dump(y_test,'y_test.pkl')
joblib.dump(lr,'lr.pkl')


X_train_resampled = joblib.load('X_train_resampled.pkl')
X_test_encoded = joblib.load('X_test_encoded.pkl')
y_train_resampled = joblib.load('y_train_resampled.pkl')
y_test = joblib.load('y_test.pkl')
lr = joblib.load('lr.pkl')

# Making predictions
predictions = lr.predict(X_test_encoded)
"""

with open ("model.py",'w') as f:
    f.write(code)


# In[ ]:




