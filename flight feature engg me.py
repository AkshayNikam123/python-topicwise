#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#to visualize all the columns in the datset
pd.pandas.set_option("display.max_columns",None)


# In[3]:


dataset=pd.read_excel(r"C:\Users\akshay\Documents\Data_Train.xlsx")


# In[4]:


dataset.head()


# In[5]:


dataset.info()


# In[10]:


dataset['Date']=dataset['Date_of_Journey'].str.split('/').str[0]
dataset['Month']=dataset['Date_of_Journey'].str.split('/').str[1]
dataset['Year']=dataset['Date_of_Journey'].str.split('/').str[2]


# In[9]:


dataset['Date']


# In[11]:


dataset['Month']


# In[12]:


dataset['Year']


# In[17]:


#split using  lambda function
dataset['Day']=dataset['Date_of_Journey'].apply(lambda x:x.split('/')[0])
dataset['Month']=dataset['Date_of_Journey'].apply(lambda x:x.split('/')[1])
dataset['Year']=dataset['Date_of_Journey'].apply(lambda x:x.split('/')[2])


# In[19]:


dataset['Day']


# In[18]:


dataset['Year']


# In[20]:


dataset.head(2)


# In[21]:


dataset['Date']=dataset['Date'].astype(int)
dataset['Month']=dataset['Date'].astype(int)
dataset['Year']=dataset['Date'].astype(int)


# In[24]:


dataset.info()


# In[26]:


dataset.drop('Date_of_Journey',axis=1,inplace=True)


# In[28]:


dataset.head(2)


# In[29]:


dataset['Arrival_Time'].str.split(' ').str[0]


# In[50]:


dataset['Dept_hour']=dataset['Dep_Time'].str.split(':').str[0]
dataset['Dept_min']=dataset['Dep_Time'].str.split(':').str[1]
dataset['Dept_hour']=dataset['Dept_hour'].astype(int)
dataset['Dept_min']=dataset['Dept_min'].astype(int)
dataset.drop('Dep_Time',axis=1,inplace=True)


# In[51]:


dataset.head(2)


# In[52]:


dataset.info()


# In[53]:


dataset['Total_Stops'].unique()


# In[56]:


dataset['Total_Stops'].mode()


# In[63]:


#why nan=1 bcoz mode is 1
dataset['Total_Stops']=dataset['Total_Stops'].map({'non-stop':0,'1 stop':1,'2 stops':1,'3 stops':1,'4 stops':1,'nan':1})


# In[64]:


dataset.head(2)


# In[62]:


dataset.drop('Route',axis=1,inplace=True)


# In[65]:


dataset.head(2)


# In[69]:


dataset['Airline'].unique()


# In[70]:


dataset['Destination'].unique()


# In[80]:


dataset=pd.get_dummies(dataset,columns=['Airline','Source','Destination','Additional_Info'],drop_first=True)


# In[ ]:





# In[81]:


dataset


# In[ ]:





# In[ ]:




