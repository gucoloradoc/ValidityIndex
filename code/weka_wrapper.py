import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.filters import Filter
jvm.start(system_cp=True, packages=True, max_heap_size="512m") #Remember to initialize this CLASSPATH env var

#Preparing the data
loader = Loader(classname="weka.core.converters.ArffLoader")
#data = loader.load_file('data/datatobayes.arff')
data = loader.load_file('data/Full.arff')
remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "first"])
remove.inputformat(data)
filtered= remove.filter(data)

#Classifier test
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
filtered.class_is_last()
classifier = Classifier(classname="weka.classifiers.bayes.BayesNet", options=['-D']) #
evaluation = Evaluation(filtered)
evaluation.crossvalidate_model(classifier, filtered, 10, Random(42))
evaluation.area_under_roc(class_index=0) #ROC
print(evaluation.summary())

#classifier.options to obtain the parameters of the classifier

jvm.stop()