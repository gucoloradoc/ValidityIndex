import pandas as pd
import numpy as np
import sys
sys.path.append('code/') #Folder with the scripts, working directory the main directory
from VIC_fun import VIC #Our created VIC function

#EXAMPLE: Creation of the clusters to validate
dataset=pd.read_csv('data/FULL_DATABASE.csv').drop(columns='Unnamed: 0')
dataset.insert(column='cluster_id', value=0, loc=len(dataset.columns))

##Running the validity index in 50 different partitions
np.random.seed(2500)
kgroups=10 #
n_jobs=4 #Multithread parameter
r=np.random.choice(range(kgroups+1,199-kgroups),50, replace=False) #Generates an array 

#Calling my classifiers, and setting their parameters
classifiers_parameters={
    'naive_bayes':{},
    'RandomForest':{
        'n_estimators':100
    },
    'svm':{
        'kernel':'linear',
        'C':1,
        'degree':2,
        'gamma':'auto'
    },
    'BayesianNet':{

    },
    'LDA':{
        
    }
}
classifiers=['svm','naive_bayes','LDA','RandomForest']
results_VIC=[] #Empyty list to store the results

for cut in r:
    dataset.loc[:,'cluster_id'].iloc[:(cut+1)]=-1
    dataset.loc[:,'cluster_id'].iloc[(cut+1):]=1
    X=dataset.iloc[:,2:-1].values
    y=dataset.iloc[:,-1].values
    missing_indexes= [(i[0],i[1])for i in np.argwhere(np.isnan(X))]
    for mis in missing_indexes:
        X[mis]=0 #Los unicos missing son numeros de documentos, por tanto quiere decir que no hay documentos registrados

    np.random.seed(1000)
    train_indexes=np.random.choice(range(200),200, replace=False)
    X_train=X[train_indexes]
    y_train=y[train_indexes]
    temp=[item for item in VIC(X_train,y_train,classifiers,kgroups,metric='roc_auc',n_jobs=n_jobs,**classifiers_parameters)]
    temp.insert(0,cut)
    results_VIC.append(temp)

