from sklearn.ensemble import RandomForestRegressor
import csv

#Read Training data
csv_reader = csv.reader(open('sample_data/co.txt', 'rU'), delimiter='\t')#delimiter="\t"
data = [x for x in csv_reader]


target = [x[0] for x in data]
train = [x[1:5] for x in data]
#create and train the random forest
#multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
rg = RandomForestRegressor(n_estimators=100, n_jobs=4)
rg.fit(train, target)

#Return Prediction
return prediction = rg.predict( )#TO BE PREDICTED
