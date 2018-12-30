---
layout: post
published: false
title: Boston HMDA Example
---
	
    #Import HDAMA file
	import pandas as pd
	csv = pd.read_csv('C:/Users/dhoic/Desktop/Hdma.csv')
	df = pd.DataFrame(csv)
    df.info()
    
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
    
    
    
