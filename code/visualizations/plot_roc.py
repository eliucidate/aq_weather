"""
This file is to plot the ROC curves of models with different feature sets.
Reference code: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
"""

import numpy as np
import matplotlib as mpl
mpl.use('Qt4Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn import linear_model
from scipy import interp
import csv
from copy import deepcopy

# functions
def calculate_accuracy(predictions, real):
	correct = 0
	for i in range(len(predictions)):
		if round(predictions[i]) == round(real[i]):
			correct += 1
	return correct * 1.0 / len(predictions)


# Import data
f = open("sample_data/pm25.csv", "rb")
reader = csv.reader(f)
X = []
y = []

for row in reader:        	
	yvar = float(row[3])
	if yvar > 0:                		
		if yvar > 12.0:
			y.append(1)
		else:
			y.append(0)
		Xvar = []
		for i in range(len(row)):
			if i == 5:
				if row[i] == 'mostlycloudy':
					Xvar.append(1.0)
				elif row[i] == 'partlycloudy':
					Xvar.append(2.0)
				elif row[i] == 'fog':
					Xvar.append(3.0)
				elif row[i] == 'clear':
					Xvar.append(4.0)
				elif row[i] == 'cloudy':
					Xvar.append(5.0)
				elif row[i] == 'rain':
					Xvar.append(6.0)
				elif row[i] == 'tstorms':
					Xvar.append(7.0)
				elif row[i] == 'hazy':
					Xvar.append(8.0)
				else:
					Xvar.append(9.0)
			elif i == 6 or i == 7 or i == 8 or i == 10 or i == 12:
				Xvar.append(float(row[i]))
			elif i == 9:
				if row[i] == "-9999.0":
					Xvar.append(0.0)
				else:
					Xvar.append(float(row[i]))    
		X.append(Xvar)

# normalize
minVar = []
maxVar = []
for val in X[0]:
	minVar.append(val)
	maxVar.append(val)
for row in X:
	for j in range(len(row)):
		if row[j] < minVar[j]:
			minVar[j] = row[j]
		if row[j] > maxVar[j]:
			maxVar[j] = row[j]
for i in range(len(X)):
	for j in range(len(X[i])):
		X[i][j] = (X[i][j] - minVar[j]) * 1.0 / (maxVar[j] - minVar[j])
		
# convert to matrix
X = np.asmatrix(X)
y = np.asarray(y)

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,
                                                    random_state=0)
clf = linear_model.LogisticRegressionCV(penalty='l2', class_weight='balanced', intercept_scaling=1e3, cv=5)

#######################################
##########    7 features      #########
#######################################
# Learn to predict each class against the other
y_score = clf.fit(X_train, y_train).decision_function(X_test)

# Calculate accuracy	
predictions = clf.predict(X)
accuracy = calculate_accuracy(predictions, y)
print "7 features accuracy: ", accuracy * 100

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr[0], tpr[0], _ = roc_curve(y_test[:], y_score[:])
roc_auc[0] = auc(fpr[0], tpr[0])

#######################################
##########    6 features      #########
#######################################
y_score_1 = clf.fit(X_train[:, [0,1,2,4,5,6]], y_train).decision_function(X_test[:, [0,1,2,4,5,6]])

predictions_1 = clf.predict(X[:, [0,1,2,4,5,6]])
accuracy_1 = calculate_accuracy(predictions_1, y)
print "6 features accuracy: ", accuracy_1 * 100

fpr[1], tpr[1], _ = roc_curve(y_test, y_score_1)
roc_auc[1] = auc(fpr[1], tpr[1])


#######################################
##########    3 features      #########
#######################################

y_score_2 = clf.fit(X_train[:, [0,3,4]], y_train).decision_function(X_test[:, [0,3,4]])

predictions_2 = clf.predict(X[:, [0,3,4]])
accuracy_2 = calculate_accuracy(predictions_2, y)
print "3 features accuracy: ", accuracy_2 * 100

fpr[2], tpr[2], _ = roc_curve(y_test, y_score_2)
roc_auc[2] = auc(fpr[2], tpr[2])

#######################################
##########    2 features      #########
#######################################

y_score_3 = clf.fit(X_train[:, [0,3]], y_train).decision_function(X_test[:, [0,3]])

predictions_3 = clf.predict(X[:, [0,3]])
accuracy_3 = calculate_accuracy(predictions_3, y)
print "2 features accuracy: ", accuracy_3 * 100

fpr[3], tpr[3], _ = roc_curve(y_test, y_score_3)
roc_auc[3] = auc(fpr[3], tpr[3])

#############################################
##########    1 features  (icon)    #########
#############################################
y_score_4 = clf.fit(X_train[:, :1], y_train).decision_function(X_test[:, :1])

predictions_4 = clf.predict(X[:, :1])
accuracy_4 = calculate_accuracy(predictions_4, y)
print "1 features (icon) accuracy: ", accuracy_4 * 100

fpr[4], tpr[4], _ = roc_curve(y_test, y_score_4)
roc_auc[4] = auc(fpr[4], tpr[4])

###################################################
##########    1 features  (wind_speed)    #########
###################################################
y_score_5 = clf.fit(X_train[:, [3]], y_train).decision_function(X_test[:, [3]])

predictions_5 = clf.predict(X[:, [3]])
accuracy_5 = calculate_accuracy(predictions_5, y)
print "1 features (wind_speed) accuracy: ", accuracy_5 * 100

fpr[5], tpr[5], _ = roc_curve(y_test, y_score_5)
roc_auc[5] = auc(fpr[5], tpr[5])

###################################################
##########    1 features  (wind_gust)    #########
###################################################
y_score_6 = clf.fit(X_train[:, [4]], y_train).decision_function(X_test[:, [4]])

predictions_6 = clf.predict(X[:, [4]])
accuracy_6 = calculate_accuracy(predictions_6, y)
print "1 features (wind_gust) accuracy: ", accuracy_6 * 100

fpr[6], tpr[6], _ = roc_curve(y_test, y_score_6)
roc_auc[6] = auc(fpr[6], tpr[6])

###################################################
##########    1 features  (temp)    #########
###################################################
y_score_7 = clf.fit(X_train[:, [1]], y_train).decision_function(X_test[:, [1]])

predictions_7 = clf.predict(X[:, [1]])
accuracy_7 = calculate_accuracy(predictions_7, y)
print "1 features (temp) accuracy: ", accuracy_7 * 100

fpr[7], tpr[7], _ = roc_curve(y_test, y_score_7)
roc_auc[7] = auc(fpr[7], tpr[7])

#########################################################
##########    1 features  (wind_speed, square)    #######
#########################################################
X_sq_train = deepcopy(X_train[:, [3]])
X_sq_train = np.append(X_sq_train, np.square(X_sq_train), axis=1)
X_sq_test = deepcopy(X_test[:, [3]])
X_sq_test = np.append(X_sq_test, np.square(X_sq_test), axis=1)

y_score_8 = clf.fit(X_sq_train, y_train).decision_function(X_sq_test)

X_sq_total = deepcopy(X[:,[3]])
X_sq_total = np.append(X_sq_total, np.square(X_sq_total), axis=1)

predictions_8 = clf.predict(X_sq_total)
accuracy_8 = calculate_accuracy(predictions_8, y)
print "1 features (wind_speed) square polynomial accuracy: ", accuracy_8 * 100

fpr[8], tpr[8], _ = roc_curve(y_test, y_score_8)
roc_auc[8] = auc(fpr[8], tpr[8])


#########################################################
##########    1 features  (wind_speed, cubic)    #######
#########################################################
temp = deepcopy(X_train[:, [3]])
X_cub_train = deepcopy(X_train[:, [3]])
X_cub_train = np.append(X_cub_train, np.square(X_cub_train), axis=1)
X_cub_train = np.append(X_cub_train, np.power(temp, 3), axis=1)

temp = deepcopy(X_test[:, [3]])
X_cub_test = deepcopy(X_test[:, [3]])
X_cub_test = np.append(X_cub_test, np.square(X_cub_test), axis=1)
X_cub_test = np.append(X_cub_test, np.power(temp, 3), axis=1)

y_score_9 = clf.fit(X_cub_train, y_train).decision_function(X_cub_test)

temp = deepcopy(X[:,[3]])
X_cub_total = deepcopy(X[:,[3]])
X_cub_total = np.append(X_cub_total, np.square(X_cub_total), axis=1)
X_cub_total = np.append(X_cub_total, np.power(temp, 3), axis=1)


predictions_9 = clf.predict(X_cub_total)
accuracy_9 = calculate_accuracy(predictions_9, y)
print "1 features (wind_speed) cubic polynomial accuracy: ", accuracy_9 * 100

fpr[9], tpr[9], _ = roc_curve(y_test, y_score_9)
roc_auc[9] = auc(fpr[9], tpr[9])

##############################################################################

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr[0], tpr[0], label='7 features (area = %0.2f)' % roc_auc[0])
plt.plot(fpr[1], tpr[1], label='6 features (area = %0.2f)' % roc_auc[1])
plt.plot(fpr[2], tpr[2], label='3 features (area = %0.2f)' % roc_auc[2])
plt.plot(fpr[3], tpr[3], label='2 features (area = %0.2f)' % roc_auc[3])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for logistic regression')
plt.legend(loc="lower right")
plt.show()

plt.figure()
plt.plot(fpr[4], tpr[4], label='icon feature (area = %0.2f)' % roc_auc[4])
plt.plot(fpr[6], tpr[6], label='wind_gust feature (area = %0.2f)' % roc_auc[6])
plt.plot(fpr[7], tpr[7], label='temperature feature (area = %0.2f)' % roc_auc[7])
plt.plot(fpr[5], tpr[5], label='wind_speed feature (area = %0.2f)' % roc_auc[5])
plt.plot(fpr[8], tpr[8], label='wind_speed square (area = %0.2f)' % roc_auc[8])
plt.plot(fpr[9], tpr[9], label='wind_speed cubic (area = %0.2f)' % roc_auc[9])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for logistic regression')
plt.legend(loc="lower right")
plt.show()


##############################################################################