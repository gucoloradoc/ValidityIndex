import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

#base_df= pd.read_csv('data/all_data.csv')
#base_df['attribute']=base_df['attribute'].astype('category')
#unive_ranking=pd.read_csv('data/search_keys2.csv')

##General distributions, all data
#attributes=base_df['attribute'].cat.categories.values
#for attribute in attributes:
#    attribute_base=base_df[base_df['attribute']==attribute]
#    ax = np.log(attribute_base.pivot(columns='year', values='count')).plot.hist(alpha=0.5,bins=50)
#    ax.set_xlabel('ln(Documents count)')
#    ax.set_ylabel('Frequency')
#    ax.set_title('Document count distribution for ' +attribute)
#    ax.figure.savefig('images/exploratory/'+attribute+'_log.pdf')

## Plotting the results of VIC
with open("output/results_VIC.list",'rb') as file:
    results_VIC=pickle.load(file)
import matplotlib.pyplot as plt
x_ax=[i[0] for i in results_VIC]
y_ax=[i[1] for i in results_VIC]
error_ax=[i[2][1] for i in results_VIC]
plt.errorbar(x_ax,y_ax,yerr=error_ax,linestyle='',marker='o', markerfacecolor='orange')
#plt.plot(x_ax,y_ax,'bo')
plt.ylabel('ROC-AUC')
plt.xlabel('r')
states=[]
