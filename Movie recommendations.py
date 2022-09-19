#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install nltk')


# In[2]:


import numpy as np # linear algebra
import pandas as pd


# In[8]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[10]:


movies = movies.merge(credits,on='title')


# In[11]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[12]:


import ast


# In[13]:


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 


# In[14]:


movies.dropna(inplace=True)
movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[15]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[16]:


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 


# In[17]:


movies['cast'] = movies['cast'].apply(convert)
movies.head()


# In[18]:


movies['cast'] = movies['cast'].apply(lambda x:x[0:3])


# In[19]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[20]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[21]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[22]:


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[23]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
#new.head()
new['tags'] = new['tags'].apply(lambda x: " ".join(x))


# In[24]:



from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
    
vector = cv.fit_transform(new['tags']).toarray()
vector.shape
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)


# In[25]:


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)


# In[26]:


import pickle
pickle.dump(new,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




