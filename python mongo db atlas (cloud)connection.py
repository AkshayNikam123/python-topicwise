#!/usr/bin/env python
# coding: utf-8

# In[1]:



1252


# In[17]:


get_ipython().system('pip install pymysql')


# In[10]:


import pymongo
client = pymongo.MongoClient("mongodb+srv://AkshayNikam:Akshay1996@cluster0.ilwkrru.mongodb.net/?retryWrites=true&w=majority") #give password in url copied from mongodb
db = client.test
print(db)


# In[11]:


database=client['akshay1']#database-bunch of collection


# In[12]:


coll=database['info']#collection-bunch of document


# In[13]:


#document
data={"name":"akshaynikam",
      "age":27,
      "dob":"03/03/1996"
     }


# In[14]:


coll.insert_one(data) #insert data mongodb


# In[15]:


many_data=[{"name":"akshaynikam",
      "age":27,
      "dob":"03/03/1996"
     },{"name":"akshaynikam",
      "age":27,
      "dob":"03/03/1996"
     },{"name":"akshaynikam",
      "age":27,
      "dob":"03/03/1996"
     },{"name":"akshaynikam",
      "age":27,
      "dob":"03/03/1996"
     }]


# In[16]:


coll.insert_many(many_data)


# # python to mongodb compass
# 

# In[2]:


import pymongo


# In[17]:


client=pymongo.MongoClient('mongodb://localhost:27017') #url copied from mongocompass


# In[13]:


mydatabase=client['sunday']


# In[14]:


a=mydatabase.employeeinfo


# In[64]:


records=[{'firstname':'akshay0',
    'lastname':'nikam','age':7,'qualification':'pahilipass'},
     {'firstname':'akshay1',
    'lastname':'nikam','age':17,'qualification':'12thpass'},
     {'firstname':'akshay2',
    'lastname':'nikam','age':27,'qualification':'graduate'}
]


# In[65]:


a.insert_many(records)


# In[66]:


mydatabase.list_collection_names()


# In[67]:


#simple way of querying


# In[68]:


a.find_one()  #top 1 records


# In[69]:


for record in a.find({}):
    print(record)
    ##similar to select * frm in sql


# In[70]:


for record in a.find({'firstname':'akshay0'}):
    print(record)


# In[72]:


#query documents using query opeartors
for record in a.find({'qualification':{'$in':['12thpass','graduate']}}):
    print(record)


# In[75]:


for record in a.find({'qualification':'graduate','age':{'$lt':28}}):
    print(record)


# In[77]:


for record in a.find({'$or':[{'qualification':'graduate'},{'age':{'$lt':28}}]}):    #or operation 
    print(record) 


# In[78]:


b=mydatabase.inventory


# In[84]:


b.insert_many([{'item':'journal','qty':25,'size':{'h':14,'w':21}},
    {'item':'notebook','qty':50,'size':{'h':8.5,'w':11}},
     {'item':'paper','qty':100,'size':{'h':8.5,'w':11}},
      {'item':'planner','qty':75,'size':{'h':22.85,'w':30}},
       {'item':'postcard','qty':45,'size':{'h':10,'w':15.25}}
])


# In[86]:


for records in b.find({'size':{'h':14,'w':21}}):
                       print(records)


# # updating json documents

# In[98]:


b.update_many({'item':'paper'},{'$set':{'item':'sketch'},'$currentDate':{'lastmodified':True}})


# In[101]:


b.replace_one({'item':'notebook'},{'item':'notebook','instock':[{'warehouse':'A','qty':60},{'warehouse':'B','qty':50}]})


# # mongodb aggreagte and group

# In[102]:


mydb=client['students']


# In[103]:


a=mydb.studentscore


# In[105]:


records=[{'user':'krish','subject':'database','score':80},
         {'user':'amit','subject':'js','score':90},
         {'user':'amit','title':'database','score':85},
         {'user':'krish','title':'js','score':75},
         {'user':'amit','title':'datascience','score':60},
        {'user':'krish','title':'data science','score':95}]


# In[106]:


a.insert_many(records)


# In[109]:


agg_result=a.aggregate(
[{
   "$group":
    {"_id":"$user",  #_id holds record in db so always use it to aggreagte
     "total_subject":{"$sum":1}
        }}
  ])
for i in agg_result:
    print(i)


# In[110]:


agg_result=a.aggregate(
[{
   "$group":   #if need to find for group
    {"_id":"$user",  #_id holds record in db so always use it to aggreagte
     "total_subject":{"$sum":"$score"}
        }}
  ])
for i in agg_result:
    print(i)


# In[111]:


agg_result=a.aggregate(
[{
   "$group":   #if need to find for group
    {"_id":"$user",  #_id holds record in db so always use it to aggreagte
     "total_subject":{"$avg":"$score"}
        }}
  ])
for i in agg_result:
    print(i)


# In[112]:


#datetime
import datetime


# In[113]:


data=[{"_id":1,"item":"abc","price":10,"qty":2,"date":datetime.datetime.utcnow()},
     {"_id":2,"item":"jkl","price":20,"qty":1,"date":datetime.datetime.utcnow()},
     {"_id":3,"item":"xyz","price":5,"qty":5,"date":datetime.datetime.utcnow()},
     {"_id":4,"item":"abc","price":10,"qty":10,"date":datetime.datetime.utcnow()},
     {"_id":5,"item":"xyz","price":5,"qty":10,"date":datetime.datetime.utcnow()}]


# In[114]:


df=mydb.stores


# In[115]:


df.insert_many(data)


# In[126]:


agg_result=df.aggregate([
        {
    '$group':{
        "_id":"$item",
        "avg_amount":{"$avg":{"$multiply":["$price","$qty"]}},
        "avgqty":{"$avg":"$qty"}
        }
        }
])
for i in agg_result:
    print(i)


# In[134]:


for i in df.aggregate([{"$project":{"_id":1,"qty":1}}]):  #project gives values id:1 means select id and qty:1 means select qty
    print(i)


# In[ ]:




