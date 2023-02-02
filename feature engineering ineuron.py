#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#to visualize all the columns in the datset
pd.pandas.set_option("display.max_columns",None)


# In[6]:


dataset=pd.read_csv(r"C:\Users\akshay\Downloads\train (3).csv")


# In[8]:


dataset.head()


# In[15]:


#x=dataset.iloc[:,1:81]
#y=dataset.iloc[:,-1]
#or

x=dataset.drop(["Id","SalePrice"],axis=1)
y=dataset["SalePrice"]


# In[16]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)


# In[17]:


x_train


# In[25]:


#missing values
feature_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>=1]
for feature in feature_nan:
    print("the feature is {} and missing value is {}%".format(feature,np.round(dataset[feature].isnull().mean(),4)))


# In[28]:


#replacing categorical feature
categorical_feature=[feature for feature in dataset.columns if dataset[feature].dtypes=='O']


# In[29]:


categorical_feature


# In[31]:


#replace categorical fetaure
data=dataset.copy()
def replcae_cate_feature(data,categorical_feature):
    data[categorical_feature]=data[categorical_feature].fillna('Missing')
    return data


# In[33]:


replcae_cate_feature(data,categorical_feature)


# In[34]:


#seond way for categorical feature
data=dataset.copy()
def replcae_cate_feature(data,categorical_feature):
    data[categorical_feature]=data[categorical_feature].fillna(data[categorical_feature].mode())
    return data


# In[36]:


replcae_cate_feature(data,categorical_feature)


# In[39]:


numerical_features_nan=[feature for feature in dataset.columns if dataset[feature].dtypes!='O'and dataset[feature].isnull().sum()>=1]


# In[40]:


numerical_features_nan


# In[42]:


#replacing null values for numerical feature
for feature in numerical_features_nan:
    ##since they are outlier we are going to replace with median
    median_values=dataset[feature].median()
    ##create a feature to capture nan values
    dataset[feature+'nan']=np.where(dataset[feature].isnull(),1,0)
    dataset[feature].fillna(median_values,inplace=True)


# In[43]:


dataset


# In[47]:


## temporal variable(datetime variable)

for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
    dataset[feature+'age of house']=dataset['YrSold']-dataset[feature]


# In[48]:


dataset 


# In[ ]:




