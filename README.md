# VIC Validity index 
This repository describes the use and implementation of internal cluster validation by using ensemble supervised classifiers, as reported in [1]. Internal metrics provide a uselful method to evaluate how appropiate is the
division expresed in a class atribbute **y** given or found (clustering), with a finite set of attributes **X** of the clustered instances, without comparing them against an external body of data.

```python
from VIC_fun import VIC
VIC(X,y,classifiers,kgroups,metric='roc_auc',n_jobs=n_jobs,**classifiers_parameters)
```

The VIC function works by an ensemble usage of five supervised learning algorithms, ***Linear Discriminant Analysis***, ***Support vector machines***, ***Random Forest***, ***Naive Bayes*** and ***Bayesian Networks***. Those algorithms are implemented using the library *scikit-learn* in python and can be passed to the VIC function as a list of strings as follows:
```python
classifiers=['svm','naive_bayes','LDA','RandomForest','BayesianNet']
```
The BayesianNet classifier is implemented using the *python-weka-wrapper3*, thus to implement this classifier in your evaluation you need to properly configure the environment to use javabridge.
In order to pass the hyperparameters to each of the classifiers, they shall be passed in a dict format with the following sintaxis:

```python
classifiers_parameters={
    'classifier_name':{
        'param_1':param_value,
        'param_2':param_value2,
        ... #more params for classifier
    },
    'classifier_2':{
        'par_1':p_1,
        'par_2':p_2,
        ... #More parameters for this classifier
    },
    ... #More classifiers
}
```
And the parameters for each of them can be found in the sklearn documentation.

K fold cross-validation is used, defining the parameter **kgroups**, to determine the best performing algorithm on a partition evaluation and its implementation can be threaded using the paramater **n_jobs** in VIC. Finally we can choose from different metrics to perform the validation through the **metric** parameter, such detailed options can be found [here](https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics).
Calling the function generates a tuple with three outputs, the maximum value of the k-fold mean metric for the evaluated classifiers, an array with ``['mean_kfold_metric', 'sd_metric', 'classifier_name' ]`` of the best classifier for the partition and a matrix with all the values for all the classifiers in case it is required.
```python
max_value, ['mean_kfold_metric', 'sd_metric', 'classifier_name' ], matrix= VIC(X,y,...)
```
## Example: Best division for 200 top QS universities using VIC


![ROC-AUC for example](images/VIC_results.png)

|r   | max ROC-AUC   |Max Classifier|
|--- | --------  |------------|
| 16 | 0.935819  |RandomForest|
|117 | 0.801157  |RandomForest|
|112 | 0.791204  |RandomForest|
|102 | 0.827702  |RandomForest|
| 80 | 0.88427   |RandomForest|
|121 | 0.81254   |RandomForest|
| 20 | 0.932244  |RandomForest|
| 77 | 0.84715   |RandomForest|
| 72 | 0.855678  |RandomForest|
|116 | 0.800921  |RandomForest|
|111 | 0.797706  |RandomForest|
| 61 | 0.86825   |RandomForest|
| 26 | 0.894063  |RandomForest|
|188 | 0.79152   |svm|
|100 | 0.843091  |svm|
| 35 | 0.894133  |RandomForest|
| 90 | 0.837121  |RandomForest|
|168 | 0.761642  |RandomForest|
| 27 | 0.902505  |LDA|
| 86 | 0.832828  |RandomForest|
|136 | 0.811571  |RandomForest|
|131 | 0.820984  |RandomForest|
|113 | 0.786027  |RandomForest|
|166 | 0.751471  |RandomForest|
|104 | 0.822475  |RandomForest|
|145 | 0.810698  |RandomForest|
| 71 | 0.869986  |RandomForest|
|144 | 0.830635  |RandomForest|
| 43 | 0.92651   |RandomForest|
|127 | 0.824565  |RandomForest|
|181 | 0.71345   |svm         |
| 41 | 0.911615  |RandomForest|
|179 | 0.686111  |svm         |
| 21 | 0.940387  |RandomForest|
| 55 | 0.864246  |RandomForest|
| 93 | 0.821465  |RandomForest|
|160 | 0.714338  |RandomForest|
|161 | 0.730852  |RandomForest|
|141 | 0.835595  |RandomForest|
| 84 | 0.860143  |RandomForest|
| 48 | 0.911083  |RandomForest|
| 52 | 0.902238  |RandomForest|
|110 | 0.793035  |RandomForest|
| 56 | 0.872183  |RandomForest|
| 24 | 0.896514  |RandomForest|
|119 | 0.804688  |RandomForest|
|176 | 0.718246  |RandomForest|
| 70 | 0.885554  |RandomForest|
|175 | 0.723557  |RandomForest|
| 81 | 0.879493  |RandomForest|
---  --------  ------------

[1] Rodríguez, J., Medina-Pérez, M. A., Gutierrez-Rodríguez, A. E., Monroy, R., & Terashima-Marín, H. (2018). Cluster validation using an ensemble of supervised classifiers. Knowledge-Based Systems, 145, 134–144. https://doi.org/10.1016/j.knosys.2018.01.010.
