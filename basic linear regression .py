#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv(r"C:\Users\akshay\Downloads\advertising (1)FSDS bootcamp2.0.csv")


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.tail()


# In[7]:


pip install ydata_profiling


# In[42]:


from ydata_profiling import ProfileReport


# In[43]:


pf=ProfileReport(df)


# In[44]:


pf


# In[46]:


pf.to_widgets()  #to get in good format


# In[50]:


pf.to_file('pandasprofilereport.html')


# In[12]:


x=df[['TV']]


# In[13]:


x


# In[14]:


y=df[['Sales']]


# In[15]:


y


# In[16]:


from sklearn.linear_model import LinearRegression


# In[28]:


linear=LinearRegression()


# In[29]:


linear.fit(x,y)


# In[30]:


linear.intercept_  #y=mx+c  c=


# In[31]:


linear.coef_   #m=


# In[32]:


file="linear_reg.sav"
pickle.dump(linear,open(file,'wb'))   #dump linear model into file


# In[33]:


linear.predict([[45]])


# In[34]:


l=[23,46,78,98,43] #calculating/predicting for n values 


# In[37]:


for i in l:
    print(linear.predict([[i]]))


# In[41]:


#load creatd model
saved_model=pickle.load(open(file,'rb'))


# In[39]:


#how to use created model
saved_model.predict([[46]])


# In[40]:


#accuracy
linear.score(x,y)


# In[ ]:




