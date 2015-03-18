#read the traning data from csv file
import csv
import numpy
import sklearn
from sklearn import svm, datasets, cross_validation
"""scikit-learn classifiers and cross validation utils
 from sklearn.ensemble import RandomForestClassifier (A random forest is a meta estimator
 that fits a number of decision tree classifiers on various sub-samples of the dataset and use
 averaging to improve the predictive accuracy and control over-fitting.)"""
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV #for cross validation
import matplotlib.pyplot as plt


#loading the training data
X=[] #traning data
Y=[] #classes

digits=datasets.load_digits()

images_and_labels=list(zip(digits.images,digits.target))

for index,(image,label) in enumerate(images_and_labels[:4]):
    plt.subplot(2,4,index+1)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    
    plt.title('Training: %i' % label)
    plt.savefig('test_image.png')


#flatten the images i.e., to turn the image into matrix
n_samples = len(digits.images)
data=digits.images.reshape((n_samples,-1)) #reshape function definition doubtful
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(data[:3*n_samples / 4], digits.target[:3*n_samples / 4])

# Now predict the value of the digit on the second half:
expected = digits.target[3*n_samples / 4:]
predicted = classifier.predict(data[3*n_samples / 4:])
print "here is what we expected"
print expected
print
print
print
print "here is what we got"
print predicted



