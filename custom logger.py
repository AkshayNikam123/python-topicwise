#!/usr/bin/env python
# coding: utf-8

# In[8]:


import logging


# In[10]:


#create a custom logger
logger=logging.getLogger(__name__)


# In[11]:


#create handler
c_handler=logging.StreamHandler()
f_handler=logging.FileHandler('abc.log')
c_handler.setLevel(logging.WARNING)
f_handler.setLevel(logging.ERROR)


# In[12]:


#create formatter
c_format=logging.Formatter('%(name)s-%(levelname)s-%(message)s')
f_format=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')


# In[14]:


c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)


# In[16]:


#add handler to logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

logger.warning('this is warning')
logger.error('this is error')


# In[33]:


#how to use in e.g


# In[17]:


logger=logging.getLogger(__name__)


# In[18]:


handler=logging.StreamHandler()


# In[19]:


format=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')


# In[20]:


handler.setFormatter(format)


# In[21]:


logger.addHandler(handler)


# In[22]:


logger.setLevel(logging.INFO)


# In[26]:


def divide(dividend,divisor):
    try:
        logger.info(f"dividing {dividend} by {divisor}")
        return dividend/divisor
    except Exception as e:
        logger.info(e)


# In[27]:


print(divide(6,2))


# In[28]:


print(divide(6,0))


# In[32]:


logging.exception("Division by zero")


# In[ ]:




