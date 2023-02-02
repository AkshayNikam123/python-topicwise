#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


df=pd.read_html(r'https://en.wikipedia.org/wiki/Football')


# In[5]:


#readable format
type(df)


# In[14]:


df1=df[0]


# In[7]:


type(df[0])


# In[15]:


df1.columns


# In[16]:


df1.head()


# In[21]:


df1.to_csv("football.csv",header=False )


# In[22]:


ls


# In[23]:


df2=df[2]


# In[24]:


df[2] #second table on website


# In[25]:


df2.to_csv('football1.csv') #creating file of data in ur pc


# In[26]:


ls


# In[35]:


d="""{"name":"Akshay ",
"email_id":"nikam.ak152@gmail.com",
"education":"Btech",
"platform":["techneuron","kidneuron","ineuron"]
}"""


# In[36]:


import json


# In[37]:


a=json.loads(d)


# In[38]:


a


# In[39]:


type(a)


# In[42]:


pd.DataFrame(a)


# In[47]:


pd.DataFrame(a['platform'])


# In[49]:


pd.DataFrame(a['education']) #cant convert dataframe to string


# In[51]:


a['education']


# In[52]:


url='https://api.github.com/repos/pandas-dev/pandas/issues'


# In[53]:


pd.read_json('https://api.github.com/repos/pandas-dev/pandas/issues')


# In[54]:


import requests
url='https://api.github.com/repos/pandas-dev/pandas/issues'


# In[55]:


data=requests.get(url)


# In[56]:


data1=data.json()


# In[57]:


data1


# In[58]:


len(data1)


# In[63]:


for i in range(len(data1)):
    print(data1[i]['user']['id'])


# In[67]:


pd.DataFrame(data1,columns=['url','repository_url','events_url','id'])


# In[ ]:




