# PCA + Clustering
Alex Papa 13/01/2020

Toy project that applies PCA for dimensionality reduction and follows through with K-means clustering on the principal components. Visualisations are built using a web-based app based on Plotly - Dash.

#### __PCA and Clustering__
[x] [Example 1. PCA + Clustering.ipynb](link)
  - My answers to the tutorial questions.
[x] [Example 1. Feature Variance PCA.ipynb](link)
  - The same example focusing on associating principal component variability to base features.
[x] Steps.txt
  - Rough outline of steps to follow.

#### __Dash Visualisation__
[ ] [Example 2. Dash.ipynb](link)
  Use simplified cancer dataset example for web app.
  - [x] 2D Interactive visualisation on Dash.
  - [x] 3D Interactive visualisation on Dash.
  - [x] 2D Scatter plot for first two principal components.
  - [x] 3D Scatter plot for first three principal components.
  - [x] PCA & Clustering 3D plot (notebook).
  - [x] K-means performance statistics (notebook).


  - [ ] Display explained variance contribution by feature.
  - [ ] ~~Offer this option instead of class in DASH drawdown.~~ (See issue)
  - [ ] K-means performance statistics in DASH.

  - [ ] Create Git repo and host app through Heroku


#### __R Shiny Implimentation__
[ ] The folder contains scripts that try to use the processed data from python and build a R web-based app using Shiny (Instead of Dash-Python)



#### __Issues__

__Dash Visualisation__
- Plotly colour coding samples
- *Doubt that 3 PCA axis visualisation is correct implimentation, maybe 2PCA and returns ?*
__PCA__
- Can calculate eigenvectors for class, but not the individual ev for a data point.


__R Shiny Implimentation__
- Complicated to integrate both R and Python environments in a single script.
-


#### Additional Ideas
