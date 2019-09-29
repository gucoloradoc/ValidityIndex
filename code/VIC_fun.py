import pandas as pd
import numpy as np
import threading
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDAclass
from sklearn.ensemble import RandomForestClassifier
#from pomegranate import * #For Bayesian Networks
from sklearn.model_selection import cross_val_score
from sklearn import metrics

def VIC(X,y,classifiers,kgroups,metric='roc_auc',n_jobs=None, **kwargs):
    """VIC is a function to calculate the validity index for the y target, usign the
    specified classifier. The kwargs are the corresponding parameters required for 
    each of the classifiers.
    """
    #Filtrar los parametros por clasificador
    if isinstance(classifiers,str):
        classifiers=[classifiers]
    clfs=[]
    scores_list=[]
    out=[]
    for classifier in classifiers:
        if classifier=='svm':
            clf = svm.SVC(**kwargs['svm'])
        elif classifier=='naive_bayes':
            clf=GaussianNB(**kwargs['naive_bayes'])
        elif classifier=='LDA':
            clf=LDAclass(**kwargs['LDA']) #Check the colinearity warnings
        elif classifier=='RandomForest':
            #Do smething
            clf=RandomForestClassifier(**kwargs['RandomForest'])
            #print('Classifier not available')
            #return None
        elif classifier=='BayesianNet':
            #Do something
            print('Trabajando para agregar Bayesian net')
            from weka_wrapper import weka_bayesnet #Calling my defined function
            bayes_out=weka_bayesnet()
            out.append([bayes_out,0,'BayesianNet'])
            continue
        else:
            raise NameError('Classifier '+classifier+' not available for VIC, or check the spelling')
        clf.fit(X,y)
        clfs.append(clf)
        scores = cross_val_score(clf, X, y, cv=int(kgroups), scoring=metric,n_jobs=n_jobs)
        print("Accuracy for "+classifier +": {} (+/- {})".format(round(scores.mean(),ndigits=3), round(scores.std(),ndigits=3)))
        scores_list.append(scores)
        out.append([scores.mean(),scores.std(), classifier])
    max_score=max([i[0] for i in out])
    max_idx=[i[0] for i in out].index(max([i[0] for i in out]))
    return max_score,out[max_idx],out
