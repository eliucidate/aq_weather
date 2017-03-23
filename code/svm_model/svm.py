
print(__doc__)
# from __future__ import division
import sys
import csv
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import cross_validation
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVR
from sklearn import svm


def load_file(file_path):
    values = []
    texts = []
    with open(file_path, 'r') as file_reader:
        reader = csv.reader(file_reader)
        for row in reader:
            # 
            value = [row[6],row[7],row[8],row[10],row[11]]
            text = row[3]
            values.append(value)
            texts.append(text)
    return normalize(values), np.array(texts)

def normalize(X) :
	X = np.array(X)
	return (preprocessing.scale(X))
def transformtolabel(x,classname):
    y = []
    x = map(float,x)
    ave = sum(x)/len(x)
    for i in range(0,len(x)):
        if float(x[i]) > ave:
            # print x[i]
            y.append(0)
        else: y.append(1)
    return y





def main():
    X,y = load_file(file_name)
    X = list(X)
    n = len(X)
    k = 5
    num = n/5
    ylabel = transformtolabel(y,k)
    score = {}
    model = {}
    
    for i in range(0,k):
        Xt = X[i*num:] +X[-(4-i)*num:]
        ylabelt = ylabel[i*num:] +ylabel[-(4-i)*num:]
        indexes = list(range(i*num,n-(4-i)*num))
        print (i*num,n-(4-i)*num)
        Xtest = [X[x] for x in indexes]
        ytest = [ylabel[i] for i in indexes]
        for kernel in ('linear','poly','rbf'):
                # print kernel
            model[kernel] = svm.SVC(kernel = kernel)
                # print kernel
            model[kernel].fit(Xt,ylabelt)
            
            score[kernel] = model[kernel].score(X,ylabel)
        print score
    #################
    # Print Boundary#
    #################
    # fignum = 1
    # for key,value in model.iteritems():
    #     clf = model[key]
    #     plt.figure(fignum,figsize = (4,3))
    #     plt.clf()
    #     plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=80,facecolor='none',zorder=2)
    #     plt.scatter(X[:,0],X[:,1],c=ylabel,zorder=2,cmap=plt.cm.Paired)
    #     plt.axis('tight')
    #     x_min = -3
    #     x_max = 3
    #     y_min = -3
    #     y_max = 3
    #     XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    #     Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    #     Z = Z.reshape(XX.shape)
    #     plt.figure(fignum, figsize=(4, 3))
    #     plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    #     plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
    #             levels=[-.5, 0, .5])

    #     plt.xlim(x_min, x_max)
    #     plt.ylim(y_min, y_max)

    #     plt.xticks(())
    #     plt.yticks(())
    #     fignum = fignum + 1
    # plt.show()
 #    model = NuSVR(C=1,nu=1)
 #    model.fit(X,y)
 #    print model.score(X,y)
 #    model = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
 #    model = NuSVR(c=1,nu=1)
 #    model.fit(Xtrain,ytrain)
 #    print model.predict(Xtrain)
	# print model.score(X,y)


if __name__ == '__main__':
    file_name = sys.argv[1]
    main()