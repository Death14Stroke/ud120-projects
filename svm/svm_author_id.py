#!/usr/bin/python3

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""

import sys
from time import time

sys.path.append("../tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

### make sure you use // when dividing for integer division


# TODO: mini project
#########################################################
### your code goes here ###

# TODO: reduce training set for faster training
# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

# TODO: change kernel to rbf and try various values for C
from sklearn.svm import SVC
clf = SVC(kernel="rbf", C=10000)
# TODO: calculate training time
t0 = time()
clf.fit(features_train, labels_train)
print("training time = ", round(time()-t0, 3), "s")

# TODO: calculate prediction time
t0 = time()
pred = clf.predict(features_test)
print("prediction time = ", round(time()-t0, 3), "s")

# TODO: calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test)
print("accuracy = ", accuracy)

print('pred for 10 = ', pred[10])
print('pred for 26 = ', pred[26])
print('pred for 50 = ', pred[50])

print('No of events predicted as Chris(1) = ', list(pred).count(1))

#########################################################
