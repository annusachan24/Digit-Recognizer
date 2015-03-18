import csv
import numpy
import sklearn
from sklearn.externals import joblib
from sklearn import svm, datasets, cross_validation
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV #for cross validation


X=[]
Y=[]

with open('train.csv','r') as traindatafile:
    traindatareader =csv.reader(traindatafile)
    for line in traindatareader:
        X.append(line[1:])
        Y.append(line[0])


X=X[1:10000]
Y=Y[1:10000]

print "hey RAM"
clf=svm.SVC(gamma=0.001,C=10)
clf.fit(X,Y)
print "haan Narayan"

with open('test.csv','r') as testdatafile:
    testdatareader = csv.reader(testdatafile)
    i=1
    for row in testdatareader:
        if i!=1 and i<=10:
            k=clf.predict(row)
            print "predicting for line" + str(i)
            print
            print k
        i+=1
