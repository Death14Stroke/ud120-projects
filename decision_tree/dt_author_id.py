#!/usr/bin/python3

""" 
    this is the code to accompany the Lesson 3 (decision tree) mini-project

    use an DT to identify emails from the Enron corpus by their authors
    
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

# TODO: mini-project
#########################################################
### your code goes here ###

# TODO: number of features
print("number of features =", len(features_train[0]))

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

# TODO: calculate accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(pred, labels_test)
print("accuracy = ", round(accuracy, 3))
#########################################################
