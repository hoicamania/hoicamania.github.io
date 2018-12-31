---
layout: post
published: false
title: Boston HMDA Example
---
My first exposure to classification was in a econometrics class where we learned about logistic regression.  The classic case was to use mortgage lending data in Boston with a goal of determining if race played a major factor in loan approvals.  The original paper found [here](https://www.bostonfed.org/publications/research-department-working-paper/1992/mortgage-lending-in-boston-interpreting-hmda-data.aspx) fostered a lot of debate, and still continues to generate buzz to this day. 

It also is a good of example of the delta between how econommists, statisticians, and computer scientists approach analyzing a problem.  Munell et al. used logistic regression to analyze key variables, but does not approach the analysis from a machine learning perspective.  Instead her focus is more on using the coefficients from the logistic regression to answer her research question.

This analysis starts out by evluating the HDMA data using logistic regression but seeking if the results still hold on test data. 

    #Import HMDA file
	import pandas as pd
	csv = pd.read_csv('C:/Users/dhoic/Desktop/Hdma.csv')
	df = pd.DataFrame(csv)
    df.info()
   

RangeIndex: 2381 entries, 0 to 2380
Data columns (total 20 columns):
Unnamed: 0     2381 non-null int64
dir            2381 non-null float64
hir            2381 non-null float64
lvr            2381 non-null float64
ccs            2381 non-null float64
mcs            2381 non-null float64
pbcr           2380 non-null object
dmi            2381 non-null object
self           2380 non-null object
single         2381 non-null object
uria           2381 non-null float64
comdominiom    2381 non-null int64
black          2381 non-null object
deny           2381 non-null object
black_yes      2381 non-null int32
deny_yes       2381 non-null int32
pbcr_yes       2381 non-null int32
dmi_yes        2381 non-null int32
self_yes       2381 non-null int32
single_yes     2381 non-null int32
dtypes: float64(6), int32(6), int64(2), object(6)
memory usage: 316.3+ KB
    
    #Clean data and create flags
    import numpy as np
	df['black_yes'] = np.where(df['black'].str.contains('no'), 0, 1)
	df['deny_yes'] = np.where(df['deny'].str.contains('no'), 0, 1)
	df['pbcr_yes'] = np.where(df['pbcr'].str.contains('no'), 0, 1)
	df['dmi_yes'] = np.where(df['dmi'].str.contains('no'), 0, 1)
	df['self_yes'] = np.where(df['self'].str.contains('no'), 0, 1)
	df['single_yes'] = np.where(df['single'].str.contains('no'), 0, 1)
	df_1 = df.drop('Unnamed: 0',axis=1)
           
    #Check for missing or NAs
    sns.heatmap(df_1.isnull(),cbar=False,yticklabels=False,cmap = 'viridis')
	plt.show()
    
    #Correlation matrix
	k = 13 #number of variables for heatmap
	cols = df_1.corr().nlargest(k, 'deny_yes')['deny_yes'].index
	cm = df_1[cols].corr()
	plt.figure(figsize=(10,6))
	sns.heatmap(cm, annot=True, cmap = 'viridis')
	plt.show()
    
    from sklearn.linear_model import LogisticRegression

	X = df_1.drop(['deny_yes','pbcr','dmi','self','single','black','deny'], axis=1)
	y = df['deny_yes']
    
    from sklearn import datasets, linear_model
	from sklearn.model_selection import train_test_split
	from matplotlib import pyplot as plt

	# create training and testing vars
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	X_train = pd.DataFrame(X_train)
	y_train = pd.DataFrame(y_train)
	X_test = pd.DataFrame(X_test)
	y_test = pd.DataFrame(y_test)
    
    from sklearn.feature_selection import RFE
	from sklearn.linear_model import LogisticRegression
	logreg = LogisticRegression()
	rfe = RFE(logreg, 20)
	rfe = rfe.fit(X_train, y_train)
	print(rfe.support_)
	print(rfe.ranking_)
    
    import statsmodels.formula.api as sm
	#features = ['lvr', 'ccs', 'mcs', 'uria', 'black_yes','pbcr_yes', 'dmi_yes']
	features = ['lvr', 'ccs', 'mcs',  'black_yes','pbcr_yes', 'dmi_yes']
	X_train_new = X_train[features]
	model = sm.Logit(y_train, X_train_new)
 	result = model.fit()
	result.summary()
    
 
	#features = ['lvr', 'ccs', 'mcs', 'uria', 'black_yes','pbcr_yes', 'dmi_yes']
	features = ['lvr', 'ccs', 'mcs',  'black_yes','pbcr_yes', 'dmi_yes']
	X_test_new = X_test[features]
	# predict class labels for the test set
	predicted = model.predict(X_test_new)
    print(metrics.confusion_matrix(y_test, predicted))
	print(metrics.classification_report(y_test, predicted))

    
    
