# PCA & t-SNE & Clustering
Alex Papa 13/01/2020

Toy project that applies PCA for dimensionality reduction and follows through with K-means / t-SNE clustering on the principal components. Visualisations are built using a web-based app based on Plotly - Dash.

#### __PCA and Clustering__
[x] [Tutorial. PCA & Clustering](https://github.com/09acp/PCA-Clustering---Dash/blob/master/Example%201.%20Feature%20Variance%20PCA.ipynb)
  - My answers to the tutorial questions.
[x] [Example 1. Feature Variance PCA](https://github.com/09acp/PCA-Clustering---Dash/blob/master/Example%201.%20Feature%20Variance%20PCA.ipynb)
  - The same example focusing on associating principal component variability to base features.
[x] Steps.txt
  - Rough outline of steps to follow.

#### __Dash Visualisation__
[x] [Example 2. Dash](https://github.com/09acp/PCA-Clustering---Dash/blob/master/Example%202.%20Dash%20-%20Plotly%20Express.ipynb)
  Use simplified cancer dataset example for web app.
  - [x] 2D Interactive visualisation on Dash.
  - [x] 3D Interactive visualisation on Dash.
  - [x] 2D Scatter plot for first two principal components.
  - [x] 3D Scatter plot for first three principal components.
  - [x] PCA & Clustering 3D plot (notebook).
  - [x] K-means performance statistics (notebook).

[x] [Example 3. PCA - Statsmodel & Dash](https://github.com/09acp/PCA-Clustering---Dash/blob/master/Example%203.%20PCA%20-%20Statsmodel%20%26%20Dash.ipynb)
  - [x] Run PCA with Statsmodel
  - [x] Compare to Sklearn PCA
  - [x] 3D Scatter plot for first three principal components.
  - [x] Display explained variance contribution by feature instead of class in DASH drawdown.
  - [x] Use t-SNE on PCs
    ![PCA explained variance by element](https://github.com/09acp/PCA-Clustering---Dash/blob/master/screenshots/pca_scatter3d.PNG?raw=true "PCA explained variance by element")

  - [x] 3D Scatter plot for t-SNE factors.
    ![t-SNE explained variance by element](https://github.com/09acp/PCA-Clustering---Dash/blob/master/screenshots/t-SNE_scatter3d.PNG?raw=true "t-SNE explained variance by element")

#### __Server Hosting__
  [x] Create Git repo
  [ ] Host app through Heroku



#### ~~__R Shiny Implimentation__~~
~~[ ] The folder contains scripts that try to use the processed data from python and build a R web-based app using Shiny (Instead of Dash-Python)~~

#### __Issues__

__Dash Visualisation__
- Plotly colour coding samples

__PCA__

__R Shiny Implimentation__
- Complicated to integrate both R and Python environments in a single script.
-


#### Additional Ideas
