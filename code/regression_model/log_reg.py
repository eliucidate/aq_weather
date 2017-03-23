'''
This file implements the logistic regression model for the relationship between 
air quality and weather features. The SGD is implemented in the gradient_descent_logreg 
function. In addition, the logistic regression model in the sklearn package and statsmodel 
package are also used.
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
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

p = 7

class Logistic_Regression:
        def __init__(self,trainingFile,testingFile,alpha,lamda,numIter,threshold):
                self.trainingFile = trainingFile
                self.testingFile = testingFile
                self.alpha = alpha
                self.lamda = lamda
                self.numIter = numIter
                self.threshold = threshold
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
                	yvar = float(row[3])
                	if yvar > 0:                		
                		if yvar > 12.0:
                			self.ytrain.append(1)
                		else:
                			self.ytrain.append(0)
                		Xvar = []
                		for i in range(len(row)):
                			if i == 6 or i == 7 or i == 8 or i == 10 or i == 12:
                				Xvar.append(float(row[i]))
                				#Xvar.append(float(row[i]) * float(row[i]))
                				#Xvar.append(float(row[i]) * float(row[i]) * float(row[i]))
                			elif i == 9:
                				if row[i] == "-9999.0":
                					Xvar.append(0.0)
                					#Xvar.append(0.0)
                					#Xvar.append(0.0)
                				else:
                					Xvar.append(float(row[i]))    
                					#Xvar.append(float(row[i]) * float(row[i]))
                					#Xvar.append(float(row[i]) * float(row[i]) * float(row[i]))
                		self.Xtrain.append(Xvar)
                #print self.Xtrain
                
                for row in testReader:                	
                	yvar = float(row[3])
                	if yvar > 0:
                		if yvar > 12.0:
                			self.ytest.append(1)
                		else:
                			self.ytest.append(0)
                		Xvar = []
                		for i in range(len(row)):
                			if i == 6 or i == 7 or i == 8 or i == 10 or i == 12:
                				Xvar.append(float(row[i]))
                				#Xvar.append(float(row[i]) * float(row[i]))
                				#Xvar.append(float(row[i]) * float(row[i]) * float(row[i]))
                			elif i == 9:
                				if row[i] == "-9999.0":
                					Xvar.append(0.0)
                					#Xvar.append(0.0)
                					#Xvar.append(0.0)
                				else:
                					Xvar.append(float(row[i])) 
                					#Xvar.append(float(row[i]) * float(row[i])) 
                					#Xvar.append(float(row[i]) * float(row[i]) * float(row[i]))                  	          			
                		self.Xtest.append(Xvar)
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
                

        def gradient_descent_logreg(self,Xtrain,ytrain):
                #TODO: Fill in the code to perform SGD to find the weights the minimize the objective function for logistic regression.
                #You might want to normalize features before you start writing SGD
                batchSize = 10
                #self.normalize(Xtrain)
                totalErr = 1
                weights = [1] * len(Xtrain[0])
                n = 0
                while n < self.numIter and totalErr > self.threshold:
                	# select a mini-batch of data
                	miniBatch = []
                	selectedIndex = []
                	for i in range(batchSize):
                		while True:
                			randomNum = random.randint(0, len(Xtrain) - 1)
                			if randomNum not in selectedIndex:
                				selectedIndex.append(randomNum)
                				break                		
                		miniBatch.append([Xtrain[randomNum], ytrain[randomNum]])
                	# calculate error
                	miniBatchErr = []
                	for i in range(batchSize):
                		miniBatchErr.append(miniBatch[i][1] - 1.0 / (1.0 + math.exp(-1.0 * self.multiply(weights, miniBatch[i][0]))))
                	totalErr = 0
                	for i in range(len(weights)):
                		sumErr = 0
                		for j in range(batchSize):
                			totalErr += abs(miniBatchErr[j])
                			sumErr += miniBatchErr[j] * miniBatch[j][0][i]
                		weights[i] += self.alpha * (sumErr - 2 * self.lamda * abs(weights[i]))
                	n += 1      	
                return weights
        
        def testGD(self,Xtest,ytest,weights):
                #TODO: Fill up the function to test your multiple regression model against the test dataset i.e. test the weights you trained by your SGD
                correct = 0
                results = []
                for i in range(len(Xtest)):
                	res = 1.0 / (1.0 + math.exp(-1.0 * self.multiply(weights, Xtest[i])))
                	results.append(round(res))
                	if round(res) == round(ytest[i]):
                		correct += 1
                accuracy = correct * 1.0 / len(ytest)
                print "test accuracy: ", accuracy * 100
                confMatrix = confusion_matrix(ytest, results, labels = [1.0, 0.0])
                print "test confusion matrix:", confMatrix
                return results
        
        def trainAccuracy(self,Xtrain,ytrain,weights):
        		correct = 0
        		results = []
        		#self.normalize(Xtrain)
        		for i in range(len(Xtrain)):
        			res = 1.0 / (1.0 + math.exp(-1.0 * self.multiply(weights, Xtrain[i])))
        			results.append(round(res))
        			if round(res) == round(ytrain[i]):
        				correct += 1
        		accuracy = correct * 1.0 / len(ytrain)
        		print "train accuracy: ", accuracy * 100
        		confMatrix = confusion_matrix(ytrain, results, labels = [1.0, 0.0])
        		print "train confusion matrix:", confMatrix
        
        def multiply(self, weights, Xtrain):
        	res = 0
        	for i in range(len(weights)):
        		res += weights[i] * Xtrain[i]
        	return res
        
        def calculate_accuracy(self, predictions, real):
        	correct = 0
        	for i in range(len(predictions)):
        		if round(predictions[i]) == round(real[i]):
        			correct += 1
        	return correct * 1.0 / len(predictions)
        	
        def calculate_MSE(self, predictions, real):
        	n = len(predictions)
        	sum = 0
        	for i in range(len(predictions)):
        		sum += (predictions[i] - real[i]) * (predictions[i] - real[i])
        	return sum * 1.0 / n
        	
               
        def sm_logit(self,Xtrain,ytrain, Xtest, ytest):
        	sm_results = sm.Logit(ytrain, Xtrain).fit_regularized(alpha = 10, disp = False)
        	print sm_results.summary()
        	# predict train labels
        	train_predictions = sm_results.predict(Xtrain)
        	train_accuracy = self.calculate_accuracy(train_predictions, ytrain)
        	print "train accuracy: ", train_accuracy * 100
        	for i in range(len(train_predictions)):
        		train_predictions[i] = round(train_predictions[i])
        	train_confMatrix = confusion_matrix(ytrain, train_predictions, labels = [1.0, 0.0])
        	print "train confusion matrix:", train_confMatrix
        	# predict test labels
        	test_predictions = sm_results.predict(Xtest)
        	test_accuracy = self.calculate_accuracy(test_predictions, ytest)
        	print "test accuracy: ", test_accuracy * 100
        	for i in range(len(test_predictions)):
        		test_predictions[i] = round(test_predictions[i])
        	test_confMatrix = confusion_matrix(ytest, test_predictions, labels = [1.0, 0.0])
        	print "test confusion matrix:", test_confMatrix
        
        def sklearn_logit(self,Xtrain,ytrain, Xtest, ytest):
        	clf = linear_model.LogisticRegressionCV(penalty='l2', class_weight='balanced', intercept_scaling=1e3, cv=5)
        	clf.fit (Xtrain, ytrain)
        	coeffients = clf.coef_
        	print "coefficients:", coeffients
        	print "intercept:", clf.intercept_
        	# predict train labels
        	train_predictions = clf.predict(Xtrain)
        	train_accuracy = self.calculate_accuracy(train_predictions, ytrain)
        	print "train accuracy: ", train_accuracy * 100
        	MSE_train = self.calculate_MSE(train_predictions, ytrain)
        	print "train MSE: ", MSE_train
        	AIC_train = len(ytrain) * np.log(MSE_train) + 2 * (p + 1)
        	print "train AIC:", AIC_train
        	for i in range(len(train_predictions)):
        		train_predictions[i] = round(train_predictions[i])
        	train_confMatrix = confusion_matrix(ytrain, train_predictions, labels = [1.0, 0.0])
        	print "train confusion matrix:", train_confMatrix
        	# predict test labels
        	test_predictions = clf.predict(Xtest)
        	test_accuracy = self.calculate_accuracy(test_predictions, ytest)
        	print "test accuracy: ", test_accuracy * 100
        	MSE_test = self.calculate_MSE(test_predictions, ytest)
        	print "test MSE: ", MSE_test
        	for i in range(len(test_predictions)):
        		test_predictions[i] = round(test_predictions[i])
        	test_confMatrix = confusion_matrix(ytest, test_predictions, labels = [1.0, 0.0])
        	print "test confusion matrix:", test_confMatrix



if __name__=='__main__':
        trainingFile = sys.argv[1]
        testingFile = sys.argv[2]
        #TODO: Make an instance of the Multiple_Regression class and pass it your the files and your parameters.
        logRegression = Logistic_Regression(trainingFile, testingFile, 0.0005, 0.0003, 65000, 0.001)
        logRegression.normalize(logRegression.Xtrain)
        logRegression.normalize(logRegression.Xtest)
        '''
        # SGD
        weights = logRegression.gradient_descent_logreg(logRegression.Xtrain, logRegression.ytrain)
        print weights
        logRegression.trainAccuracy(logRegression.Xtrain, logRegression.ytrain, weights)
        results = logRegression.testGD(logRegression.Xtest, logRegression.ytest, weights)
        with open('pm25_test_result.txt', 'wb') as wf:
        	for res in results:
        		if res == 0.0:
        			wf.write("LOW")
        		else:
        			wf.write("HIGH")
        		wf.write("\n")
        wf.close()
        '''
        logRegression.sklearn_logit(logRegression.Xtrain, logRegression.ytrain, logRegression.Xtest, logRegression.ytest)
        
        
        # Naive Bayes
        gnb = GaussianNB()
        bayesResult = gnb.fit(logRegression.Xtrain, logRegression.ytrain).predict(logRegression.Xtest)
        correct = 0
        for i in range(len(bayesResult)):
        	if round(bayesResult[i]) == round(logRegression.ytest[i]):
        		correct += 1
        accuracy = correct * 1.0 / len(logRegression.ytest)
        print "Bayes accuracy: ", accuracy * 100
        
        




















