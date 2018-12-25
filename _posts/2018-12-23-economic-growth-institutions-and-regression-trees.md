---
layout: post
published: true
title: 'Economic Growth, Institutions, and Regression Trees'
---
Economic growth that reaches all corners of society remain a common goal among governments across the world. 

Unsurpringsly, there exists thousands of studies on the correct procedure for moving economic growth forward. 

One overlooked source of economic growth is a society's dependency on governmental and non-govermental instiutional development.  Economists, Daron Acemoglu, Simon Johnson, and James A. Robinson, produced a prominent paper illustrating the importantce of institutions growing an economy using ordinary least squares regression.  Results and a link to the full paper can be found by [clicking here](https://economics.mit.edu/files/4123).  

The purpose of this analysis is to illustrate that the same results can be achieve using regression trees.

1. The first steps begins by loading in the key packages of Sci-kit Learn, Pandas, Numpy, and a few other packages.


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor  
from sklearn import metrics 

2. The second step revolves around important and cleaning the data.

df1 = pd.read_stata('https://github.com/QuantEcon/QuantEcon.lectures.code/raw/master/ols/maketable1.dta')
Dropping NA's is required to use numpy's polyfit
df1_subset = df1.dropna(subset=['logpgp95', 'avexpr','cons1','cons90','democ00a','cons00a','extmort4','logem4','loghjypl','baseco'])
df1_subset.head()
![inst gdp growth.PNG]({{site.baseurl}}/img/inst gdp growth.PNG)

3. Test out the accuracy of several different tree depths.  

y = df1_subset['logpgp95']
X = df1_subset.drop(['logpgp95','shortnam'], axis=1)

branch = [1,2,3,4,5,6,7,8,9,10,12,13,14,15]
result_array = np.array([])

for k in branch: 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=k, random_state=0)  
    regressor = DecisionTreeRegressor()
    regressor.fit(X_train, y_train) 
    y_pred = regressor.predict(X_test)  
    predict_df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred}) 
    #print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    result = metrics.mean_squared_error(y_test, y_pred)
    result_array = np.append(result_array, result)
    result_acc = result_array.tolist()
    
4.  Chose the tree depth with the lowest mean square error.

Plot the data
plt.scatter(branch, result_acc,color='darkgreen', marker='^')

Add a legend
plt.legend()

plt.xlabel('Branch')
plt.ylabel('MSE')

Show the plot
plt.show()

![MSE vs Tree Depth.PNG]({{site.baseurl}}/img/MSE vs Tree Depth.PNG)

5. Visualize the results of three with the following code

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("C:/Users/dhoic/Desktop/dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

![tree.PNG]({{site.baseurl}}/img/tree.PNG)
