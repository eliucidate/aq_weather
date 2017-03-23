'''
This file implements the multiple linear regression model for the relationship between 
air quality and weather features. The SGD is implemented in the multireg function. In 
addition, ridge model and lasso model in sklearn package, and the multiple regression 
model in statsmodel package are also used.
'''

import numpy as np
import sys
import csv
from copy import deepcopy
from sklearn import linear_model
import random
import math
from sklearn.cross_validation import cross_val_predict
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn import cross_validation

class Multiple_Regression:
        def __init__(self,trainingFile,testingFile,alpha,lamda,numIter,threshold):
                self.trainingFile = trainingFile
                self.testingFile = testingFile
                #self.alpha = alpha
                #self.lamda = lamda
                #self.numIter = numIter
                #self.threshold = threshold
                #TODO: Read in the training and testing files
                #print "************* parameters ***************"
                #print "alpha:", alpha, "lamda:", lamda, "threshold:", threshold
                trainF = open(trainingFile, "rb")
                testF = open(testingFile, "rb")
                trainReader = csv.reader(trainF)
                testReader = csv.reader(testF)
                self.Xtrain = []
                self.ytrain = []
                self.Xtest = []
                self.ytest = []
                for row in trainReader:
                	'''
                	if row[9] != "-9999.0":
                		self.ytrain.append(float(row[3]))
                		#self.Xtrain.append([float(row[8]), float(row[12]), float(row[6]), float(row[7]), float(row[10]), float(row[9])])
                		self.Xtrain.append([float(row[9])])
                		#print float(row[3]), float(row[9])
                	'''

                	self.ytrain.append(float(row[3]))
                	
                	Xvar = []
                	for i in range(len(row)):
                		if i == 6 or i == 7 or i == 8 or i == 10 or i == 12:
                			Xvar.append(float(row[i]))
                		elif i == 9:
                			if row[i] == "-9999.0":
                				Xvar.append(0.0)
                			else:
                				Xvar.append(float(row[i]))    
                	          			
                	self.Xtrain.append(Xvar)
                print "median is", np.median(self.ytrain)
                #print self.Xtrain
                
                for row in testReader:
                	'''
                	if row[9] != "-9999.0":
                		self.ytest.append(float(row[3]))
                		#self.Xtest.append([float(row[8]), float(row[12]), float(row[6]), float(row[7]), float(row[10]), float(row[9])])
                		self.Xtest.append([float(row[9])])
                	'''
                	self.ytest.append(float(row[3]))
                	Xvar = []
                	for i in range(len(row)):                	
                		if i == 6 or i == 7 or i == 8 or i == 10 or i == 12:
                			Xvar.append(float(row[i]))
                		elif i == 9:
                			if row[i] == "-9999.0":
                				Xvar.append(0.0)
                			else:
                				Xvar.append(float(row[i]))
                	               			
                	self.Xtest.append(Xvar)
                # find the minVar and maxVar for the training data
                self.minVar = []
                self.maxVar = []
                for val in self.Xtrain[0]:
                	self.minVar.append(val)
                	self.maxVar.append(val)
                for row in self.Xtrain:
                	for j in range(len(row)):
                		if row[j] < self.minVar[j]:
                			self.minVar[j] = row[j]
                		if row[j] > self.maxVar[j]:
                			self.maxVar[j] = row[j]

        def normalize(self,X):
                #TODO: Fill in the code to normalize features!
                for i in range(len(X)):
                	for j in range(len(X[i])):
                		X[i][j] = (X[i][j] - self.minVar[j]) * 1.0 / (self.maxVar[j] - self.minVar[j])
                

        def multireg(self,Xtrain,ytrain, Xtest, ytest):    
                self.normalize(Xtrain)
                '''
                # polynomial try
                poly = PolynomialFeatures(degree=2)
                Xtrain = poly.fit_transform(Xtrain)
                Xtest = poly.fit_transform(Xtest)
                '''
                # normal clf fit
                clf = linear_model.LinearRegression()
                clf.fit (Xtrain, ytrain)
                coeffients = clf.coef_
                print "coefficients:", coeffients
                print "intercept:", clf.intercept_
                
                print "train score", clf.score(Xtrain,ytrain)
                print "test score", clf.score(Xtest,ytest)
                # manual calculate train accuracy
                train_results = clf.predict(Xtrain)
                print "first x:", Xtrain[0]
                print "first result:", train_results[0]
                correct = 0
                for i in range(len(train_results)):
                	if round(train_results[i], 1) == round(ytrain[i], 1):
                		correct += 1
                accuracy = correct * 1.0 / len(ytrain)
                print "train accuracy: ", accuracy * 100, "%"
                # cross validation
                score = cross_validation.cross_val_score(clf, Xtrain, ytrain, scoring='mean_squared_error', cv = 5)
                print "cross validation score: ", score
                
                predict = cross_val_predict(clf, Xtrain, ytrain, cv = 5)
                correct = 0
                for i in range(len(predict)):
                	if round(predict[i], 1) == round(ytrain[i], 1):
                		correct += 1
                accuracy = correct * 1.0 / len(ytrain)
                print "cross validation accuracy: ", accuracy * 100, "%"
                # manual calculate test accuracy
                self.normalize(Xtest)
                results = clf.predict(Xtest)
                correct = 0
                for i in range(len(results)):
                	if round(results[i], 1) == round(ytest[i], 1):
                		correct += 1
                accuracy = correct * 1.0 / len(ytest)
                print "test accuracy: ", accuracy * 100, "%"
                
                return coeffients
        
        def ridge_multireg(self,Xtrain,ytrain, Xtest, ytest):    
                self.normalize(Xtrain)
                '''
                # polynomial try
                poly = PolynomialFeatures(degree=2)
                Xtrain = poly.fit_transform(Xtrain)
                Xtest = poly.fit_transform(Xtest)
                '''
                # normal clf try
                clf = linear_model.Ridge(alpha = 10000)
                clf.fit (Xtrain, ytrain)
                coeffients = clf.coef_
                print "train score", clf.score(Xtrain,ytrain)
                print "test score", clf.score(Xtest,ytest)
                # manual calculate train accuracy
                train_results = clf.predict(Xtrain)
                correct = 0
                for i in range(len(train_results)):
                	if round(train_results[i], 1) == round(ytrain[i], 1):
                		correct += 1
                accuracy = correct * 1.0 / len(ytrain)
                print "train accuracy: ", accuracy * 100, "%"
                # cross validation
                score = cross_validation.cross_val_score(clf, Xtrain, ytrain, scoring='mean_squared_error', cv = 5)
                print "cross validation score: ", score
                '''
                predict = cross_val_predict(clf, Xtrain, ytrain, cv = 5)
                correct = 0
                for i in range(len(predict)):
                	if round(predict[i]) == round(ytrain[i]):
                		correct += 1
                accuracy = correct * 1.0 / len(ytrain)
                print "cross validation accuracy: ", accuracy * 100, "%"
                '''
                # manual calculate test accuracy
                self.normalize(Xtest)
                results = clf.predict(Xtest)
                correct = 0
                for i in range(len(results)):
                	if round(results[i], 1) == round(ytest[i], 1):
                		correct += 1
                accuracy = correct * 1.0 / len(ytest)
                print "test accuracy: ", accuracy * 100, "%"
                
                return coeffients
        def lasso_multireg(self,Xtrain,ytrain, Xtest, ytest):    
                self.normalize(Xtrain)
                clf = linear_model.Lasso(alpha = 0.5)
                clf.fit (Xtrain, ytrain)
                coeffients = clf.coef_
                print "coeffients: ", coeffients
                print "train score", clf.score(Xtrain,ytrain)
                print "test score", clf.score(Xtest,ytest)
                # manual calculate train accuracy
                train_results = clf.predict(Xtrain)
                correct = 0
                for i in range(len(train_results)):
                	if round(train_results[i], 1) == round(ytrain[i], 1):
                		correct += 1
                accuracy = correct * 1.0 / len(ytrain)
                print "train accuracy: ", accuracy * 100, "%"
                # cross validation
                predict = cross_val_predict(clf, Xtrain, ytrain, cv = 5)
                correct = 0
                for i in range(len(predict)):
                	if round(predict[i], 1) == round(ytrain[i], 1):
                		correct += 1
                accuracy = correct * 1.0 / len(ytrain)
                print "cross validation accuracy: ", accuracy * 100, "%"
                
                # manual calculate test accuracy
                self.normalize(Xtest)
                results = clf.predict(Xtest)
                correct = 0
                for i in range(len(results)):
                	#print round(results[i], 1), round(ytest[i], 1)
                	if round(results[i], 1) == round(ytest[i], 1):
                		correct += 1
                accuracy = correct * 1.0 / len(ytest)
                print "test accuracy: ", accuracy * 100, "%"
                
                return coeffients
                
        def sm_multireg(self,Xtrain,ytrain, Xtest, ytest):    
                self.normalize(Xtrain)
                results = sm.OLS(ytrain, Xtrain).fit_regularized()
                #print "coefficient: ", results.params
                # train accuracy
                predictions = results.predict(Xtrain)
                correct = 0
                for i in range(len(predictions)):
                	if round(predictions[i], 1) == ytrain[i]:
                		correct += 1
                accuracy = correct * 1.0 / len(ytrain)
                print "train accuracy: ", accuracy * 100, "%"
                # calculate SSE, SSM & SST
                SSE = 0
                for i in range(len(predictions)):
                	SSE += (predictions[i] - ytrain[i])**2
                yAverage = np.mean(ytrain)
                SSM = 0
                for pred in predictions:
                	SSM += (pred - yAverage)**2
                print "SSM:", SSM
                SST = SSE + SSM
                print "SST:", SST
                # calculate PVE = SSM / SST
                PVE = SSM / SST
                print "PVE:", PVE
                # test accuracy
                self.normalize(Xtest)
                predictions = results.predict(Xtest)
                correct = 0
                for i in range(len(predictions)):
                	print round(predictions[i], 1), ytest[i]
                	if round(predictions[i], 1) == ytest[i]:
                		correct += 1
                accuracy = correct * 1.0 / len(ytest)
                print "test accuracy: ", accuracy * 100, "%"
                return results
        
        
        def multiply(self, weights, Xtrain):
        	res = 0
        	for i in range(len(weights)):
        		res += weights[i] * Xtrain[i]
        	return res
        	
        
        	



if __name__=='__main__':
        trainingFile = sys.argv[1]
        testingFile = sys.argv[2]
        #TODO: Make an instance of the Multiple_Regression class and pass it your the files and your parameters.
        mRegression = Multiple_Regression(trainingFile, testingFile, 0.001, 0.01, 250000, 0.1)
        weights = mRegression.multireg(mRegression.Xtrain, mRegression.ytrain, mRegression.Xtest, mRegression.ytest)
        #print weights.summary()
        
        '''
        results = mRegression.testGD(mRegression.Xtest, mRegression.ytest, weights)
        with open('../data/quality.txt', 'wb') as wf:
        	for res in results:
        		wf.write(str(res))
        		wf.write("\n")
        wf.close() 
        '''




















