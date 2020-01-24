#!/usr/bin/env python
# coding: utf-8

# # 3D Visualisation (Plotly/Dash)  
# __Cancer Dataset__
# 
# *To be run with the anaconda MIX environment for both Python and R*
# 
# - https://towardsdatascience.com/dive-into-pca-principal-component-analysis-with-python-43ded13ead21
# - 30 features, 569 samples, 2 classes

# In[1]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm_notebook
from datetime import datetime

# VISUALIZATIONS
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
# import plotly.graph_objs as go
import plotly.express as px

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
plt.style.use(['seaborn-darkgrid'])
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['font.family'] = 'DejaVu Sans'

from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
RANDOM_STATE = 17 # Fixed random state for reproductivity 


# In[2]:


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.data.shape


# In[3]:


cancer_df=pd.DataFrame(cancer.data,columns=cancer.feature_names)# just convert the scikit learn data-set to pandas data-frame.
plt.subplot(1,2,1)#fisrt plot
plt.scatter(cancer_df['worst symmetry'], cancer_df['worst texture'], s=cancer_df['worst area']*0.05, color='magenta', label='check', alpha=0.3)
plt.xlabel('Worst Symmetry',fontsize=12)
plt.ylabel('Worst Texture',fontsize=12)
plt.subplot(1,2,2)# 2nd plot
plt.scatter(cancer_df['mean radius'], cancer_df['mean concave points'], s=cancer_df['mean area']*0.05, color='purple', label='check', alpha=0.3)
plt.xlabel('Mean Radius',fontsize=12)
plt.ylabel('Mean Concave Points',fontsize=12)
plt.tight_layout()
plt.show()


# In[4]:


scaler=StandardScaler()#instantiate
scaler.fit(cancer.data) # compute the mean and standard which will be used in the next command
X_scaled=scaler.transform(cancer.data)# fit and transform can be applied together and I leave that for simple exercise
# we can check the minimum and maximum of the scaled features which we expect to be 0 and 1


# In[5]:


cancer.data.shape


# In[6]:


pca=PCA(n_components=3) 
pca.fit(X_scaled) 
X_pca=pca.transform(X_scaled) 
#let's check the shape of X_pca array
print ("shape of X_pca", X_pca.shape) # samples 


# In[7]:


pca.explained_variance_


# In[8]:


Xax=X_pca[:,0]
Yax=X_pca[:,1]
labels=cancer.target
cdict={0:'red',1:'green'}
labl={0:'Malignant',1:'Benign'}
marker={0:'*',1:'o'}
alpha={0:.3, 1:.5}
fig,ax=plt.subplots(figsize=(7,5))
fig.patch.set_facecolor('white')
for l in np.unique(labels):
    ix = np.where(labels==l)
    ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40,
           label=labl[l],marker=marker[l],alpha=alpha[l])
# for loop ends
plt.xlabel("First Principal Component",fontsize=14)
plt.ylabel("Second Principal Component",fontsize=14)
plt.legend()
plt.show()
# please check the scatter plot of the remaining component and you will understand the difference


# In[9]:


pca.components_.shape, X_scaled.shape


# In[10]:


plt.matshow(pca.components_,cmap='viridis')
plt.yticks([0,1,2],['1st Comp','2nd Comp','3rd Comp'],fontsize=20)
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)),cancer.feature_names,rotation=65,ha='left')
plt.tight_layout()
plt.show()# 


# In[11]:


pca.components_.shape


# In[12]:


df = pd.DataFrame(pca.components_[0], index = cancer.feature_names, columns = ['1st Principal Component'])
df.sort_values(by= ['1st Principal Component'], ascending=False );


# In[13]:


pca.components_


# # Visualisations

# #### OLD 
# https://plot.ly/python/line-and-scatter/#dash-example
fig = go.Figure(data=go.Scattergl(
    x = X_pca[:,0],
    y = X_pca[:,1],
    mode='markers',
    marker=dict(
                color= 2,
                colorscale='Viridis',
                line_width=1
                )
))
fig.show() 
# ## Test Figures
# - Medium blog: https://medium.com/plotly/introducing-plotly-express-808df010143d
# - Plotly Express: https://plot.ly/python/plotly-express/
# - Attributes for Scatter plot: https://plot.ly/python-api-reference/plotly.express.html
# 

# In[14]:


df_pca = pd.DataFrame(X_pca)
df_pca['Class'] = cancer.target # add classes
df_pca.columns = ["PCA_1","PCA_2","PCA_3","Class" ]
df_pca.head(2)


# ### PCA 
# - https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
# 
# We have found the optimal stretch and rotation in  high dimensional space that allows us to see the layout of the digits in two / three dimensions. 
# 
# 

# #### 2D Figure

# In[15]:


import plotly.express as px
fig = px.scatter(df_pca.astype(str),  # .astype(str) makes the colour map discrete
                 x="PCA_1", 
                 y="PCA_2", 
                 color="Class",
                 color_continuous_scale='Inferno',
                 hover_name= "Class",
                 title = " 2 Main Principal Components ",
                 
                )
                 #size='petal_length', hover_data=['petal_width'])
fig.show()


# #### 3D Figure

# In[16]:


fig = px.scatter_3d(df_pca.astype(str), # .astype(str) makes the colour map discrete
                 x="PCA_1", 
                 y="PCA_2",
                 z="PCA_3", 
                 color="Class",
                 color_continuous_scale='Inferno',
#                  size = 'variable',   #size='petal_length', hover_data=['petal_width'])
#                  size_max = 10,
                 hover_name= "Class",
                 title = " 3 Main Principal Components ",
                 width = 1000,
                 height = 1000,
                )
                 
fig.show()


# ## PCA & K-Means 

# In[17]:


"""
Calculating the sum of squared distances from each point to its assigned center(distortions).
"""
inertia = []

for k in tqdm_notebook(range(1, 15)): # Just 2 classes, but try up to 15
    kmeans = KMeans(n_clusters=k, n_init=100, 
                    random_state=RANDOM_STATE, n_jobs=1).fit(X_pca)
    inertia.append(np.sqrt(kmeans.inertia_))
    


# ##### _6_ clusters seems to be the ideal number of groups 

# In[18]:


# Elbow plot
plt.figure(figsize=(16,8))
plt.plot(range(1, 15), inertia, marker='s');
plt.xlabel('K clusters')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')


# #### Matplotlib Test

# In[19]:


kmeans = KMeans(n_clusters=6, 
                n_init=100, 
                random_state=RANDOM_STATE, 
                n_jobs=1)
kmeans.fit(X_pca)
cluster_labels = kmeans.labels_


# In[20]:


plt.scatter( X_pca[:, 0], X_pca[:, 1], c=cluster_labels, s=20, cmap='viridis');


# #### Impliment Clustering on 3D PCA plot 

# In[21]:


df_pca['Cluster'] = cluster_labels
df_pca.head(3)


# In[22]:


pca.components_.shape  #(3, 30)
df_pca.shape  #(569, 5)

" __ ISSUE __ "
" Can calculate eigenvectors for class, but not the individual ev for a data point "
# df_data = pd.DataFrame(cancer.data)
# df_data.columns = cancer.feature_names
# df_data.head(2)

# df_eigenvectors = pd.DataFrame(pca.components_)
# df_eigenvectors.columns = cancer.feature_names
# df_eigenvectors.head(2) 


# In[23]:


fig = px.scatter_3d(df_pca.astype(str), # .astype(str) makes the colour map discrete
                 x="PCA_1", 
                 y="PCA_2",
                 z="PCA_3", 
                 color="Cluster",
                 #color_continuous_scale='Viridis', # Inferno
                 hover_name= "Class",
                 title = " 3 Main Principal Components ",
                 width = 1000,
                 height = 1000,
                )
                 
fig.show()


# #### K-Means Performance Statistics

# In[24]:


tab = pd.crosstab( df_pca['Class'], df_pca['Cluster'], margins=True)  # data points, cluster names
tab.index = ['0', '1','2']
tab.columns = ['cluster' + str(i + 1) for i in range(6)] + ['all'] # classes
tab


# In[25]:


pca.components_.shape    , pca.explained_variance_.shape


# In[26]:


pca.explained_variance_


# In[ ]:





# ## Dash App

# In[27]:


" Create and specify html template "
app = dash.Dash()
# Boostrap CSS.
app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'}) 


# In[28]:


app.layout = html.Div([
        
    dcc.Tabs(id="tabs", children=[  # ALL TABS START

        # TAB 1. 2D Scatter Plot
        dcc.Tab(label= '2D PCA', children=[
            html.Div([
                html.H1(" Two Component Scatter Plot ", style={'textAlign': 'center'}),

                #dcc.Graph(id='plot_1'),
                dcc.Graph(figure= px.scatter(
                                     df_pca.astype(str), 
                                     x="PCA_1", 
                                     y="PCA_2", 
                                     color="Class",
                                     hover_name= "Class",
                                     title = " 2 Main Principal Components "
                                     )        
                           )    
            ]),
        ]),  # END TAB 1 
        
        # TAB 2. 3D Scatter Plot
        dcc.Tab(label= '3D PCA', children=[
            html.Div([
                html.H1(" Three Component Scatter Plot ", style={'textAlign': 'center'}),

                #dcc.Graph(id='plot_1'),
                dcc.Graph(figure =px.scatter_3d(
                                     df_pca.astype(str), # .astype(str) makes the colour map discrete
                                     x="PCA_1", 
                                     y="PCA_2",
                                     z="PCA_3", 
                                     color="Class",
                                     color_continuous_scale='Inferno',
                                     hover_name= "Class",
                                     title = " 3 Main Principal Components ",
                                     width = 1150,
                                     height = 1150,
                                    ) 
                           )   
            ]),
        ]),  # END TAB 2 
        
        
        # TAB 3. 3D Scatter Plot: PCA & Clustering
        dcc.Tab(label= '3D PCA & Clustering', children=[
            html.Div([
                html.H1(" Three Component Scatter Plot ", style={'textAlign': 'center'}),

                #dcc.Graph(id='plot_1'),
                dcc.Graph(figure = px.scatter_3d(
                                     df_pca.astype(str), # .astype(str) makes the colour map discrete
                                     x="PCA_1", 
                                     y="PCA_2",
                                     z="PCA_3", 
                                     color="Cluster",
                                     #color_continuous_scale='Viridis', # Inferno
                                     hover_name= "Class",
                                     title = " 3 Main Principal Components ",
                                     width = 1000,
                                     height = 1000,
                                    )
                           )   
            ]),
        ]),  # END TAB 3
        
   ]), # ALL TABS END  
], className="container")


# In[29]:


if __name__ == '__main__':
    app.run_server()


# ## Dash App 2 with Callbacks 

# #### Callbacks are unecessary with plotly express since figures fitted directly in app layout 
" Create and specify html template "
app = dash.Dash()
# Boostrap CSS.
app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'}) app.layout = html.Div([
        
    dcc.Tabs(id="tabs", children=[  # ALL TABS START

        # TAB 1. 2D Scatter Plot
        dcc.Tab(label= '2D PCA', children=[
            html.Div([
                html.H1(" Two Component Scatter Plot ", style={'textAlign': 'center'}),

                dcc.Graph(id = 'plot_1'),   
            ]),
        ]),  # END TAB 1 
        
        
        
        
   ]), # ALL TABS END  
], className="container")# Tab 1, 2D Scatter Plot 
@app.callback(
              Output('plot_1', 'figure')
              
             )

def scatter_2d (df_pca):   

    figure = px.scatter(df_pca, 
                     x="PCA_1", 
                     y="PCA_2", 
                     color="Class",
                     hover_name= "Class",
                     title = " 2 Main Principal Components "
                    )
    return figure
    
# RUN APP
if __name__ == '__main__':
    app.run_server()
# In[ ]:





# ================================================= END ============================================================

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




