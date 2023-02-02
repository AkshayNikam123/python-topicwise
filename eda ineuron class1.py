#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

#display all the columns of the dataframe
pd.pandas.set_option('display.max_columns',None)


# In[2]:


dataset=pd.read_csv(r"C:\Users\akshay\Downloads\train (3).csv")


# In[3]:


dataset.head()


# In[4]:


#print shape of dataset
print(dataset.shape)


# ## In data analysis we will analyse to find the below stuff
# 1.Missing values
# 2.All the numerical variables
# 3.Distribution of numerical variable
# 4.categorical variables
# 5.outliers
# 6.Realtionship bet independent and dependent features
# 7.correlation
# 

# In[7]:


#missng values
dataset.isnull().sum()


# In[13]:


#features which are having missing value
features_with_nan=[features for  features in dataset.columns if dataset[features].isnull().sum()>1] #list comprehension


# In[14]:


features_with_nan


# In[20]:


dataset['Electrical'].isnull().mean()


# In[25]:


for feature in features_with_nan:
    print(feature,np.round(dataset[feature].isnull().mean(),4)*100,'%missing values')
    # 4 is to get percentage


# In[29]:


sns.boxplot(dataset['SalePrice'])


# In[28]:


#lets plot some diagram

data=dataset.copy()
for feature in features_with_nan:
   
    #lets make a variable that indicate 1 if the observation was missing or 0 otherwise
    
    data[feature]=np.where(data[feature].isnull(),1,0)
    
   # lets calculate the mean sales price where the information is missing or present
    data.groupby(feature)['SalePrice'].median().plot.bar() #median to avoid outlier
    plt.title(feature)
    plt.show()
    #sale price with others relationship


# Here with realtion between missing value and dependent variable is clearly visible.So we need to replace thse ann values with something meaning ful which we will do in feature engg section

# # Numerical variable

# In[34]:


## list of numerical variables
dataset['SaleType'].dtypes!='object'


# In[39]:


numerical_features=[feature for feature in dataset.columns if dataset[feature].dtype!='O']


# In[40]:


print(len(numerical_features))
dataset[numerical_features].head()


# In[41]:


dataset.info()


# In[44]:


##temporal variable (e.g Datetime variables)
year_feature=[feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]


# In[45]:


for feature in year_feature:
    print(feature,dataset[feature].unique())


# In[47]:


dataset.groupby('YrSold')['SalePrice'].median().plot() #realtion bet these using median value
plt.xlabel('Year Sold')
plt.ylabel(' Median house price')
plt.title('House Price vs Year Sold')


# In[51]:


#here will compare the difference between all years features with salesprice
data=dataset.copy()
for feature in year_feature:
    if feature !='yrSold':
        data[feature]=data['YrSold']-data[feature]
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()


# In[53]:


##observations
# """"from above plot it is clearly visible new homes are costlier than older houses 
#and it shows power law distribution"""


# In[55]:


#numerical varibale types
## continuous variable and discrete variable

discrete_feature=[feature for feature in numerical_feature if len(dataset[feature].unique())<=25]
print(len(discrete_feature))


# In[56]:


dataset[discrete_feature].head()


# In[58]:


## realtion bet discrete and sales price
data=dataset.copy()
for feature in discrete_feature:
    data.groupby(feature)["SalePrice"].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()    


# In[59]:


#there is relationship between discrete feature and sale price  


# In[63]:


### continuous varibale
continuous_feature=[feature for feature in numerical_feature if feature not in discrete_feature+year_feature]


# In[64]:


print(len(continuous_feature))


# In[68]:


## lets analyse the continuous values by creating histograms to understand the distribution
data=dataset.copy()
for feature in continuous_feature:
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("count")
    plt.title(feature)
    plt.show()


# ## eda part2

# In[69]:


## we will be using logorithmc transformation
data=dataset.copy()
for feature in continuous_feature:
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.title(feature)
        plt.show()


# In[70]:


##outliers
for feature in continuous_feature:
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()


# In[71]:


sns.histplot(dataset['SalePrice'],kde=True)


# # Categorical variable

# In[74]:


categorical_feature=[feature for feature in dataset.columns if dataset[feature].dtypes=='O']


# In[76]:


dataset[categorical_feature].head()


# In[78]:


for feature in categorical_feature:
    print("the feature name is{} and the no of categories are {}".format(feature,len(dataset[feature].unique())))


# In[82]:


## find relationship bet categorical_feature with sale price
data=dataset.copy()
for feature in categorical_feature:
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# In[ ]:




