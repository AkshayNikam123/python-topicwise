#!/usr/bin/env python
# coding: utf-8

# In[108]:


get_ipython().run_line_magic('ls', '')


# In[109]:


pwd()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[110]:


f=open(r"C:\Users\akshay\Documents\akshay1.txt","rt")


# In[111]:


print(f.read())


# In[112]:


f.tell() #get cursor point on which file


# In[114]:


f.seek(0) #cursor to original or specified position


# In[34]:


z=open("akshay2.txt","w")
z.write("hello looking for something")


# In[47]:


z=open(r"akshay2.txt","w")
print(z.read())    #actual path


# In[32]:


f=open(r"C:\Users\akshay\Documents\akshay1.txt","rt")

print(f.read(5))


# In[116]:


with open(r"C:\Users\akshay\Documents\akshay1.txt","w") as f:
    f.write("new line\n")
    f.write("new line\n")
    


# In[119]:


print(f.read())# as with open closes file automatically


# In[121]:


f=open("akshay0.txt","w+")
f.write("hello")
print(f.read(4))


# In[31]:


f=open(r"C:\Users\akshay\Documents\akshay1.txt","rt")
for x in f:
    print(x)


# In[37]:


f=open(r"C:\Users\akshay\Documents\akshay1.txt","rt")
print(f.readline())
print(f.readline())


# In[38]:


f=open(r"C:\Users\akshay\Documents\akshay1.txt","rt")
print(f.readline())
print(f.readline())
f.close()


# In[40]:


f=open(r"C:\Users\akshay\Documents\akshay1.txt","a")
z=f.write("want to add more content")


# In[41]:


f=open(r"C:\Users\akshay\Documents\akshay1.txt","rt")
print(f.read())


# In[42]:


print(f.read())


# In[44]:


f=open(r"C:\Users\akshay\Documents\akshay1.txt","w")
f.write("oops i delete the content")


# In[46]:


f=open(r"C:\Users\akshay\Documents\akshay1.txt","r")
print(f.read())


# In[52]:


s=open("akshay4.txt","x")


# In[53]:


p=open(r"C:\Users\akshay\Documents\akshay1.txt","w")


# In[58]:


import os
if os.path.exists("akshay2.txt"):
    os.remove("akshay2.txt")
else:
    print("file doesnt exist")


# In[60]:


import os
if os.path.exists("akshay3.txt"):
    os.remove("akshay3.txt")
else:
    print("file doesnt exist")


# In[61]:


import os
if os.path.exists("akshay3.txt"):
    os.remove("akshay3.txt")
else:
    print("file doesnt exist")


# In[62]:


import os
if os.path.exists("akshay1.txt"):
    os.remove("akshay1.txt")
else:
    print("file doesnt exist")


# In[63]:


f=open(r"C:\Users\akshay\Documents\akshay1.txt","rt")
print(f.read())


# In[65]:


import os
if os.path.exists(r"C:\Users\akshay\Documents\akshay1.txt"):
    os.remove("akshay1.txt")
else:
    print("file doesnt exist")


# In[66]:


f=open(r"C:\Users\akshay\Documents\akshay1.txt","rt")
print(f.read())


# In[69]:


import os
if os.path.exists(r"C:\Users\akshay\Documents\akshay1.txt"):
    os.remove(r"C:\Users\akshay\Documents\akshay1.txt")
else:
    print("file doesnt exist")


# In[70]:


f=open(r"C:\Users\akshay\Documents\akshay1.txt","rt")
print(f.read())


# In[71]:


import os
if os.path.exists(r"C:\Users\akshay\Documents\akshay1.txt"):
    os.remove(r"C:\Users\akshay\Documents\akshay1.txt")
else:
    print("file doesnt exist")


# In[74]:


a=[1,2,3,4]
ans=list()
for i in a:
    print(i**2)
    ans.append(i**2)
    
    
print(ans)


# In[75]:


a=['a','a','b','c']
for i in a:
    print(i,a.count(i))


# In[78]:


a=[1,2,3]
b=[2,3,4]
A=a+b
print(A)


# In[84]:


a="akshay"
a[0:2]
print(a[-4:-1])


# In[80]:


a=[1,2,3,4,5]
a[1:4]


# In[87]:


a="jay "
a.replace('y','yyy')


# In[88]:


a.isspace()


# In[89]:


def test1():
    print("im akshay")


# In[93]:


def test18(func):
  def test19():
    print ("im inside test19")
    func()
  return test19()
def test22(*args):
  print("this is retrun of test22")


# In[94]:


test18(test22)


# In[95]:


def test6(func):
  def test7(*args,**kwargs): #args and kwargs in wrapper function compulsory 
    func(*args,**kwargs)
    print(func(*args,**kwargs))
    print("this is my decprator function")
    return(func(*args,**kwargs))
  return test7


# In[96]:


def test8():
  return 5+7


# In[97]:


@test6
def test8():
  return 5+7


# In[98]:


test8()


# In[101]:


test6(test8())


# In[104]:


print(dir(locals()['__builtins__']))


# In[107]:


x=1
y=0
assert y!=0


# In[122]:


x=('a',1,2,3,4)
for i in x:
    print(i)


# In[123]:


myiter="ineuron"
for i in myiter:
    print(i)


# In[130]:


dir(str)


# In[ ]:





# In[129]:


dir(x)


# In[ ]:




