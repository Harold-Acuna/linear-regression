#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


base=pd.read_stata("C:\\Users\\sebas\\Documentos\\Universidad\\Sexto semestre\\Econometria 2\\crime.dta")


# In[4]:


base.head()


# In[5]:


xt = np.array([np.ones(len(base)),base["polpc"]])	


# In[6]:


x= np.transpose(xt)


# In[7]:


xt_x= np.dot(xt,x)


# In[8]:


len(base)


# In[9]:


xt_x1=np.linalg.inv(xt_x)


# In[10]:


xt_y=np.dot(xt,base["crmrte"])


# In[11]:


betas= np.dot(xt_x1,xt_y)


# In[21]:


import matplotlib.pyplot as plt


# In[22]:


plt.scatter (base["polpc"],base["crmrte"])
plt.xlim (0,0.01)
m=np.linspace (0,0.01,100)
y=betas[0]+betas[1]*m
plt.plot (m,y)


# In[ ]:




