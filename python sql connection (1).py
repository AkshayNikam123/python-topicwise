#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install mysql-connector-python


# In[6]:


import mysql.connector

# Establish a connection to the MySQL server
conn = mysql.connector.connect(
    host="localhost",
    port=3306,
    user="root",  #sql user
    password="Akshay@1996", 
    database="akshaynikam"
)


# In[7]:


cursor = conn.cursor()


# In[9]:


cursor.execute("SELECT * FROM fsds")


# In[10]:


rows = cursor.fetchall()


# In[11]:


for row in rows:
    print(row)


# In[12]:


cursor.close()
conn.close()


# # OR

# In[2]:


get_ipython().system('pip install pymysql')


# In[5]:


import pymysql


# In[7]:


a=pymysql.connect(host='localhost',user='root',password='Akshay@1996',database='importdb')


# In[8]:


a


# In[9]:


import pandas as pd


# In[13]:


b=pd.read_sql_query("""select * from employee_hr_data""",a,parse_dates=None)  #sql query


# In[14]:


b.head()


# In[ ]:




