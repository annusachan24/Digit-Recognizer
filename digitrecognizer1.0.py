#read the traning data from csv file
import csv
import numpy
import sklearn
from sklearn.externals import joblib
from sklearn import svm, datasets, cross_validation
"""scikit-learn classifiers and cross validation utils
 from sklearn.ensemble import RandomForestClassifier (A random forest is a meta estimator
 that fits a number of decision tree classifiers on various sub-samples of the dataset and use
 averaging to improve the predictive accuracy and control over-fitting.)"""
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV #for cross validation



#loading the training data
X=[] #traning data
Y=[] #classes

with open('train.csv','r') as traindatafile:
    traindatareader=csv.reader(traindatafile)
    for line in traindatareader:
        #line=line.strip('\n')
        X.append(line[1:])
        Y.append(line[0])


#X=X #remove the label in first row
#Y=Y
"""print X[0:10]
print Y[0:10]"""
print "a123 check check"

clf=svm.SVC(gamma=0.128,C=1)
#clf=joblib.load('train_pickle.pkl')
clf.fit(X[1:1000],Y[1:1000])
print "testing testing"

#why whole definition of svc is not shown in interpreter
testarray=[]
"""with open('testtest.csv','r') as testdatafile:
    testdatareader=csv.reader(testdatafile)
    i=1
    for line in testdatareader:
            #print "testing the classifier "
            
            testarray.append(line[1:])"""
            
        


#print testarray
print clf.predict(X[1000:1010]) 

    
#joblib.dump(clf,'train_pickle.pkl')
