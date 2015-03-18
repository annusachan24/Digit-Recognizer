###Aaron J. Bradley
###Support Vector Machine for Handwritten Digit Recognition

import csv
import numpy as np
from sklearn import svm, datasets, cross_validation
from sklearn.grid_search import GridSearchCV

###Load Training Data
trainTargetArray = []
trainDataArray = []
with open('./train.csv', 'r') as trainFile:
	trainReader = csv.reader(trainFile, delimiter = ',')
	for row in trainReader:
		trainTargetArray.append(row[0])		
		trainDataArray.append(row[1:])

#Delete column headers		
del trainTargetArray[0]
del trainDataArray[0]
trainData = np.array(trainDataArray)
trainTarget = np.array(trainTargetArray)
trainData = trainData.astype(np.float)/255.0
trainTarget = trainTarget.astype(np.float)

###Load Testing Data
testDataArray = []
with open('./test.csv', 'r') as testFile:
	testReader = csv.reader(testFile, delimiter = ',')
	for row in testReader:
		testDataArray.append(row)

#Delete column headers		
del testDataArray[0]
testData = np.array(testDataArray)
testData = testData.astype(np.float)/255.0

#Set up classification and fit the model data
svc = svm.SVC(gamma=0.128, C=1)
svc.fit(trainData, trainTarget)
	
#Predict/Determine Value of New Images
prediction = svc.predict(testData)

#Save output to file
output = open('./output.csv', 'w')
for x, value in np.ndenumerate(prediction):
	output.write(str(int(value)))
	output.write("\n")
output.close()
