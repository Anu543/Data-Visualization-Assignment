#!/usr/bin/env python
# coding: utf-8

# # Data Visualization Assignment
# 
# - In this assignment students have to transform iris data into 3 dimensions and plot a 3d chart with transformed dimensions and colour each data point with specific class.

# In[6]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets


# # Loading the IRIS Dataset

# In[26]:


# Load the IRIS dataset
iris = datasets.load_iris()

 
# Create dataframe using iris.data
df = pd.DataFrame(data=iris.data, columns= iris.feature_names)
 
# Append class / label data
df["class"] = iris.target
 
# Print the data and check for yourself
df.head()


# In[17]:


print(iris.data.shape)


# In[19]:


print(iris.feature_names)


# In[32]:


plt.figure(figsize=(15, 4))
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'] , c=iris.target)
plt.colorbar(ticks=[0, 1, 2])
plt.title('Iris Dataset Scatterplot')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')


# In[36]:


ax = plt.axes(projection="3d")

plt.show()


# In[38]:


from sklearn.decomposition import PCA
X_reduced = PCA(n_components=3).fit_transform(iris.data)
X_reduced


# In[41]:


fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)

ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)

ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()


# In[ ]:




