#!/usr/bin/env python
# coding: utf-8

# # Analysis of people were likely to survive in Titanic shipwreck happened in      1912

# ## Problem statement
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in
# history. On April 15, 1912, during her maiden voyage, the Titanic sank after
# colliding with an iceberg, killing numerous passengers and crew. This
# sensational tragedy shocked the international community and led to better
# safety regulations for ships.
# One of the reasons that the shipwreck led to such loss of life was that there
# were not enough lifeboats for the passengers and crew. Although there was
# some element of luck involved in surviving the sinking, some groups of people
# were more likely to survive than others, such as women, children, and the
# upper-class.
# In this, we ask you to complete the analysis of what sorts of people were likely
# to survive. In particular, we ask you to apply the tools of machine learning to
# predict which passengers survived the tragedy.

# # Importing libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# # Reading the data

# In[3]:


Titanic_data = pd.read_csv(r"C:\Users\akshay\Downloads\train (1).csv")


# In[4]:


Titanic_data.head()


# In[5]:


Titanic_data.tail()


# In[5]:


Titanic_data.nunique()


# In[6]:


Titanic_data.columns


# # Exploratory data analysis and cleaning

# In[7]:


Titanic_data.describe(include="all")


# In[ ]:





# In[8]:


Titanic_data.info()


# In[9]:


Titanic_data.shape


# In[10]:


Titanic_data.isnull()


# # 1.Null values

# In[11]:


#finding out the null values in data
Titanic_data.isnull().sum()


# In[12]:


sns.heatmap(Titanic_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Nearly 20-25% of age data is missing.The proportion of age missing is likely small enough for reasonable replacement.
# Looking at cabin data,it looks like we are missing too much of the data to something useful so we'll probably drop it.

# In[13]:


#here we are figuring out avg value of age so that we can put it in missing values
plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=Titanic_data,palette='winter')


# In[14]:


# Each class has different avg value so imput values accordingly

def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age


# In[15]:


Titanic_data['Age']=Titanic_data[['Age','Pclass']].apply(impute_age,axis=1)


# In[16]:


sns.heatmap(Titanic_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[17]:


Titanic_data.drop(['Cabin','Ticket','Name','PassengerId'],axis=1,inplace=True)


# In[18]:


# Finding the mode value of "Embarked" column
print(Titanic_data["Embarked"].mode())


# In[19]:


print(Titanic_data["Embarked"].mode()[0])


# In[20]:


# Replacing the missing value in "Embarked" column with the mode value

Titanic_data["Embarked"].fillna(Titanic_data["Embarked"].mode()[0], inplace=True)


# In[21]:


Titanic_data.head()


# In[22]:


Titanic_data.dropna(inplace=True)


# In[23]:


Titanic_data.isnull().sum()


# # 2.Data Visualisation

# In[24]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=Titanic_data)


# In[25]:


# finding the number of people survived and not survived

Titanic_data['Survived'].value_counts()


# In[26]:


# making a count plot for "Sex" column 
sns.countplot ("Sex", data=Titanic_data )


# In[27]:


sns.set_style('whitegrid')
sns.countplot('Sex',hue='Survived',data=Titanic_data)


# In[28]:


#women survived more compared to men


# In[29]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=Titanic_data,palette='rainbow')

pd.crosstab(Titanic_data['Pclass'],Titanic_data['Survived']).apply(lambda r: round((r/r.sum())*100,1),axis=1)


# In[30]:


#More people survived from first class compared to second and third class


# In[31]:


sns.distplot(Titanic_data['Age'].dropna(),kde=False,color='darkred',bins=40)


# In[32]:


survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = Titanic_data[Titanic_data['Sex']=='female']
men = Titanic_data[Titanic_data['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
d= ax.set_title('Male')


# In[33]:


# Male and female between age group 15-40 likely to survive


# In[34]:


sns.countplot(x='SibSp',data=Titanic_data)


# In[35]:


#No. of people travelling alone are more


# In[36]:


Titanic_data['Fare'].hist(color='green',bins=40,figsize=(8,4))


# In[37]:


Titanic_data.hist(figsize=(20,30))


# In[38]:


sns.pairplot(Titanic_data, kind = 'reg', hue='Survived' ,size = 2)


# In[39]:


plt.figure(figsize = (20,15))
sns.heatmap(Titanic_data.corr(),annot=True)


# Fare,survival,age,sibsp,parch are fairly correlated with each other

# In[40]:


pd.pivot_table(Titanic_data, index = 'Survived', values = ['Age','SibSp','Parch','Fare'])


#   1. Average age of surviving people is 28 
#   2. people travelling with first class which is having higher fare rates likely to survive 
#   3. Childern have higher chance of survival as compared to parents as preference given to children in case life jacket
#   4. Survival of person whos having siblings spouse is less
# 

# In[41]:


sns.distplot(Titanic_data['Age'])

print(Titanic_data['Age'].skew())

print(Titanic_data['Age'].kurt())


# In[42]:


sns.distplot(Titanic_data["Fare"])


# In[43]:


#Fare data is highly skewed so it affects our conclusion so we need to remove outliers


# In[44]:


plt.figure(figsize=(12,7))
sns.boxplot(Titanic_data['Fare'])


# ## Outlier removal

# In[45]:


Q1=Titanic_data.Fare.quantile(0.25)
Q3=Titanic_data.Fare.quantile(0.75)
Q1,Q3


# In[46]:


IQR=Q3-Q1
IQR


# In[47]:


print("old shape:",Titanic_data.shape)


# In[48]:


upper_limit=Q3+1.5*IQR
lower_limit=Q1-1.5*IQR


# In[49]:


Titanic_data[(Titanic_data.Fare < lower_limit)|(Titanic_data.Fare>upper_limit)]


# In[50]:


Titanic_data[(Titanic_data.Fare >lower_limit)&(Titanic_data.Fare<upper_limit)]


# In[51]:


Titanic_data.describe()


# ## converting categorical feature 

# In[52]:


Titanic_data['Sex'].value_counts()


# In[53]:


Titanic_data['Embarked'].value_counts()


# In[54]:


a=pd.get_dummies(Titanic_data.Embarked,prefix='Embarked')
a


# In[55]:


Titanic_data=Titanic_data.join(a)
Titanic_data.drop(['Embarked'],axis=1,inplace=True)


# In[56]:


Titanic_data.head()


# In[57]:


Titanic_data.Sex=Titanic_data.Sex.map({'male':0,'female':1})


# In[58]:


Titanic_data.head()


# In[60]:


# scale continuous variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Titanic_data[['Age', 'Fare']] = scaler.fit_transform(Titanic_data[['Age', 'Fare']])


# In[61]:


Titanic_data.hist(figsize=(20,30))


# In[62]:


sns.pairplot(Titanic_data, kind = 'reg', hue='Survived' ,size = 2)


# In[63]:


# As data points are overlapping we prefer random forest algorithm as it gives good result


# ## Separating target and feature

# In[64]:


y=Titanic_data.Survived.copy()
X=Titanic_data.drop(['Survived'],axis=1)


# ## Spliting data 

# In[65]:


from sklearn.model_selection import train_test_split


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify = y, random_state=2)


# In[67]:


print(X.shape, X_train.shape, X_test.shape)


# # Model building

# ## Logistic Regression

# In[68]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# ## Train data

# In[69]:


model.fit(X_train,y_train)


# In[70]:


# accuracy on training data
X_train_prediction = model.predict(X_train)


# In[71]:


print(X_train_prediction)


# In[72]:


training_data_accuracy = accuracy_score(y_train,X_train_prediction)
print('Accuracy score of the training data : ', training_data_accuracy)


# ## Test data

# In[73]:


X_test_prediction = model.predict(X_test)


# In[74]:


print(X_test_prediction)


# In[75]:


test_data_accuracy = accuracy_score(y_test,X_test_prediction)
print('Accuracy score of the test data : ', test_data_accuracy)


# In[76]:


from sklearn.metrics import confusion_matrix


# In[77]:


accuracy=confusion_matrix(y_test,X_test_prediction)


# In[78]:


accuracy


# In[79]:


accuracy=accuracy_score(y_test,X_test_prediction)


# In[80]:


accuracy


# ## Random Forest

# In[81]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
acc_random_forest


# In[82]:


models = [LogisticRegression(), SVC(kernel='linear'), KNeighborsClassifier(), RandomForestClassifier()]


# In[83]:



models = [LogisticRegression(), SVC(kernel='linear'), KNeighborsClassifier(), RandomForestClassifier()]
def compare_models_train_test():
    for model in models:
        
        # Training the model
        model.fit(X_train, y_train)
        
        #Evaluating the model
        test_data_prediction = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, test_data_prediction)
        print('Accuracy score of the ', model, '=', accuracy)
    


# In[84]:


compare_models_train_test()


# In[85]:


cv_score_lr = cross_val_score(LogisticRegression(max_iter=1000),X,y,cv=5,)

print(cv_score_lr)

mean_accuracy_lr = sum(cv_score_lr)/len(cv_score_lr)
mean_accuracy_lr = mean_accuracy_lr*100
mean_accuracy_lr = round(mean_accuracy_lr,2)
print(mean_accuracy_lr)


# In[86]:


cv_score_SVC = cross_val_score(SVC(kernel='linear'),X,y,cv=5)

print(cv_score_SVC)

mean_accuracy_SVC = sum(cv_score_SVC)/len(cv_score_SVC)
mean_accuracy_SVC = mean_accuracy_SVC*100
mean_accuracy_SVC = round(mean_accuracy_SVC,2)
print(mean_accuracy_SVC)


# In[87]:


cv_score_KN = cross_val_score(KNeighborsClassifier(),X,y,cv=5)

print(cv_score_KN)

mean_accuracy_KN = sum(cv_score_KN)/len(cv_score_KN)
mean_accuracy_KN = mean_accuracy_KN*100
mean_accuracy_KN = round(mean_accuracy_KN,2)
print(mean_accuracy_KN)


# In[88]:


cv_score_RF = cross_val_score(RandomForestClassifier(),X,y,cv=5)

print(cv_score_RF)

mean_accuracy_RF = sum(cv_score_RF)/len(cv_score_RF)
mean_accuracy_RF = mean_accuracy_RF*100
mean_accuracy_RF = round(mean_accuracy_RF,2)
print(mean_accuracy_RF)


# ### As we can see that accuracy in random forest is more after cross validation

# In[89]:


importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)


# In[90]:


importances.plot.bar()


# ### Parch,embarked are not that significant

# In[91]:


Titanic_data.drop(['Parch','Embarked_S','Embarked_C','Embarked_Q'],axis=1,inplace=True)


# In[92]:


#Using random forest again after removal of less significant feature
random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
random_forest.fit(X_train, y_train)
Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")


# In[95]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(random_forest, X_train, y_train, cv=3)
confusion_matrix(y_train, predictions)


# In[96]:


from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(y_train, predictions))
print("Recall:",recall_score(y_train, predictions))


#  Our model predicts 75.57% of the time, a passengers survival correctly (precision). The recall tells us that it predicted the survival of 77.52 % of the people who actually survived. 

# ### 1. Both male and female from age group 15-40 likely to survive.
# ### 2. people travelling with first class which is having higher fare rates likely to survive .

# In[ ]:





# In[ ]:




