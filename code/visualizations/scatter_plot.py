#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This file is to draw the scatter plot of the data points with given two features as axes. 
Reference code: http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html 
"""

import matplotlib as mpl
mpl.use('Qt4Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import csv
import numpy as np

# import data
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
                

X = np.asmatrix(X)
y = np.asarray(y)

x_min, x_max = X[:, 6].min() - .05, X[:, 6].max() + .05
y_min, y_max = X[:, 3].min() - .05, X[:, 3].max() + .05

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 6], X[:, 3], c=y, cmap=plt.cm.Paired)
plt.xlabel('vism')
plt.ylabel('windspeed')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.show()
