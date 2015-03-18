#read the traning data from csv file
from __future__ import print_function
import csv

import numpy as np
import sklearn
from sklearn.externals import joblib
from sklearn import svm, datasets, cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
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
    traindatareader=csv.reader(traindatafile,delimiter=',')
    for line in traindatareader:
        #line=line.strip('\n')
        X.append(line[1:])
        Y.append(line[0])


X=X[1:42000] #remove the label in first row
Y=Y[1:42000]
#print X
#print
#print
#print Y

X_float=np.array(X)
Y_float=np.array(Y)
X_float=X_float.astype(np.float)/255.0
Y_float=Y_float.astype(np.float)
#print X[0:10]
#print Y[0:10]
# Split the dataset in equal parts
X_train, X_test, Y_train, Y_test = train_test_split(
    X_float, Y_float, test_size=0.25, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.125,0.1,0.01],
                     'C': [1,1.05,10,100]},
                    {'kernel': ['linear'], 'C': [1,1.05,10,100]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring=score)
    clf.fit(X_train, Y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    Y_true, Y_pred = Y_test, clf.predict(X_test)
    print(classification_report(Y_true, Y_pred))
    print()

"""print "please give me the mistake"
clf=svm.SVC(gamma=0.001,C=1.01)
clf.fit(X_train,Y_train)
print "or run correctly" """

#why whole definition of svc is not shown in interpreter
test_data=[]
with open('test.csv','r') as testdatafile:
    testdatareader=csv.reader(testdatafile,delimiter=',')
    for row in testdatareader:
        test_data.append(row)



test_data=test_data[1:]
test=np.array(test_data)
test=test.astype(np.float)/255.0
prediction=clf.predict(test)

print (prediction)

#writing the ouput to the csv file
with open('output.csv','w') as outfile:
   writer=csv.writer(outfile,delimiter=",")
   i=1
   for value in prediction:
       
        #writer.writerow(str(int(value)))   using this wrote an extra blank line in between the values of prediction
        if i==1:
            outfile.write("ImageId")
            outfile.write(",")
            outfile.write("Label")
            outfile.write("\n")
        else:
            outfile.write(str(int(i-1)))
            outfile.write(",")
            outfile.write(str(int(value)))
            outfile.write("\n")
        i+=1

    
joblib.dump(clf,'train_pickle.pkl')
