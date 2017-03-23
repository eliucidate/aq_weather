from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from numpy import genfromtxt, savetxt
from itertools import izip
from sklearn.externals import joblib
import csv

###########
# Note:
# I looked at https://www.kaggle.com/c/digit-recognizer/forums/t/2299/getting-started-python-sample-code-random-forest
###########

def grouped(iterable, n):
    return izip(*[iter(iterable)]*n)

def main():
    #create the training & test sets, skipping the header row with [1:]
    #dataset = genfromtxt(open('weatherData.tsv','r'), delimiter='\t', dtype='f8') #f8 , usecols=(0,1,2,3,4,5,6)
    #print [x for x in dataset]
    #better_data = []
    #for i0, i1, i2, i3, i4, i5, i6, i7 in grouped(dataset, 8): #lets exclude the nans.
        #better_data.append([i0, i1, i2, i3, i4, i5, i6])
    #print better_data


    csv_reader = csv.reader(open('co.txt', 'rU'), delimiter='\t')#delimiter="\t"
    data = [x for x in csv_reader]
    delim = len(data)
    test = data[-81:]
    data2 = data[:(delim-81)]
    print len(data)
    print len(data2)


    target = [x[0] for x in data2]
    print len(target)
    train = [x[1:7] for x in data2]
    print len(train)
    #create and train the random forest
    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=#)
    #rf = RandomForestClassifier(n_estimators=100, n_jobs=4)
    rg = RandomForestRegressor(n_estimators=100, n_jobs=4)
    #rf.fit(train, target)
    rg.fit(train, target)

    #test = [x[1:] for x in csv.reader(open('sample_data/co.txt', 'rU'), delimiter = '\t')]
    test1 = [x[1:7] for x in test]
    print len(test1)

    #prediction = rf.predict(test1)
    #print prediction

    prediction = rg.predict(test1)##rf

    anum = 0
    mean_sq_e = 0
    for apred in prediction:
        mean_sq_e += (float(apred)-float(test[anum][0]))**2

    data_sum = 0
    for x in data:
        data_sum += float(x[0])

    data_var = 0
    for cntr in data:
        data_var += ((float(cntr[0]) - data_sum/len(data)))**2
    print "data variance", data_var/len(data)
    print "data average", data_sum/len(data)
    print "mean_sq_e", mean_sq_e/len(data)

    print "R squared:", 1-mean_sq_e/data_var
    print "feature importance", rg.feature_importances_##rf

    with open('predsco.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(prediction)



if __name__=="__main__":
    main()
