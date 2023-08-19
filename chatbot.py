#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv("C:/Users/palla/Documents/Py files/datasets/kagglesdsdata_datasets.txt", sep="\t")


# In[3]:


data


# In[4]:


# New row data
top_row = {'hi, how are you doing?': 'hi, how are you doing?', "i'm fine. how about yourself?": "i'm fine. how about yourself?"}

# new DataFrame for the new row
row_df = pd.DataFrame([top_row])

data=pd.concat([row_df,data])
data=data.rename(columns={'hi, how are you doing?':"Ques", "i'm fine. how about yourself?":"Ans"})
data


# ## data

# In[8]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
import string



# In[9]:


def cleaner(x):
    x= x.translate(str.maketrans('', '', string.punctuation))
    x=x.lower()
    x=x.split()
    return x
#We used translate method to remove all punctuation characters, then convert to lower case and then split it into list of words


# In[10]:


pp=Pipeline([('vectorizer', CountVectorizer(analyzer=cleaner)), #CountVectorizer converts input text data into a matrix of token counts
             ('tfidf',TfidfTransformer()),
             ('classifier',DecisionTreeClassifier())])
pp.fit(data['Ques'],data['Ans'])


# In[20]:


print(pp.predict(['hi how are you doing'])[0])


# In[23]:


print(pp.predict(["what's going on"])[0])


# In[25]:


print(pp.predict(["you got a minute"])[0])

