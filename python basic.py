#!/usr/bin/env python
# coding: utf-8

# # class and instances (object)

# In[1]:


class computer:
    def config(self):
        print("i5,16gb,1TB")
        


# In[2]:



#object
com1=computer()


# In[3]:


print(type(com1))


# In[4]:


computer.config(com1)


# In[5]:


com1.config()


# In[ ]:





# In[21]:



#class creation
class employee:
    def __init__(self,first,last,pay):#attribute 
        self.first=first
        self.last=last
        self.pay=first
        self.email=first+'.'+last+'@gmail.com'
    def fullname(self):
        return '{} {}'.format(self.first,self.last)


# In[22]:


#object creation
emp1=employee('Akshay','Nikam',50000)
emp2=employee('prats','john',40000)


# emp1

# In[23]:


emp1.email  #working- class goes to object emp1 and finds email and attach argument given in it to email as defined in class


# In[24]:


emp1.fullname() #parenthisis needed as its a method(function)


# In[25]:


emp2.fullname()
#working-  class goes to object emp2 and finds method(func)i.e fullname here and attach/apply argument given in  emp2 to fullname 


# In[26]:


#can print with class name
employee.fullname(emp1)


# In[28]:


employee.fullname(emp1) #as employee doesnt undestand which data to put so it is neceesary to give object as a parameter(argument) 


# # class and variables

# In[33]:


class employee:
    raiseamount=1.04  #variable define in class
    def __init__(self,first,last,pay):#attribute 
        self.first=first
        self.last=last
        self.pay=pay
        self.email=first+'.'+last+'@gmail.com'
    def fullname(self):
        return '{} {}'.format(self.first,self.last)
    def apply_raise(self):
        self.pay = int(self.pay * self.raiseamount)


# In[34]:


emp1=employee('Akshay','Nikam',50000)
emp2=employee('prats','john',40000)


# In[35]:



emp1.apply_raise()
emp1.pay


# In[36]:


emp1.__dict__


# In[37]:


employee.raiseamount  #can call through class


# In[39]:


emp1.raiseamount #can call through object also


# In[40]:


class employee:
    raiseamount=1.04  #variable define in class
    numofemployee=0   #variable
    def __init__(self,first,last,pay):#attribute 
        self.first=first
        self.last=last
        self.pay=pay
        self.email=first+'.'+last+'@gmail.com'
        employee.numofemployee +=1 #it is given here as init runs every time
    def fullname(self):
        return '{} {}'.format(self.first,self.last)
    def apply_raise(self):
        self.pay = int(self.pay * self.raiseamount)


# In[41]:


emp1=employee('Akshay','Nikam',50000)
emp2=employee('prats','john',40000)


# In[42]:


employee.numofemployee


# In[1]:


class employee:
    raiseamount=1.04  #variable define in class
    numofemployee=0   #variable
    def __init__(self,first,last,pay):#attribute 
        self.first=first
        self.last=last
        self.pay=pay
        self.email=first+'.'+last+'@gmail.com'
        employee.numofemployee +=1 #it is given here as init runs every time
    def fullname(self):
        return '{} {}'.format(self.first,self.last)
    def apply_raise(self):
        self.pay = int(self.pay * self.raiseamount)
    def set_raise_amount(cls,amount):
        cls.set_raise_amount=amount
        


# In[2]:


emp1=employee('Akshay','Nikam',50000)
emp2=employee('prats','john',40000)


# In[6]:


emp1.set_raise_amount(1.05)


# 
