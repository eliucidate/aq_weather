# -*- coding: utf-8 -*-
"""
Created on Tue May 10 19:19:49 2016

@author: ericy
"""
import requests
import csv
from dateutil.parser import parse
import itertools

data =[]
dates = []
pol = ['pm25', 'pm10', 'so2', 'no2', 'o3', 'co', 'bc']
for i in pol:
    skip = False
    data =[]
    dates = []
    filename = i+'SP.csv'
    with open('C:\\Users\\ericy\\Documents\\cs1951\\'+filename, 'r') as csvin:
        reader = csv.reader(csvin)
        row_count = sum(1 for row in reader)
        csvin.seek(0)
        if row_count >= 40:
            avg = []
            for row in reader:
                date = row[3][:10]
                pol = row[5]
                val = row[6]
                data.append((date, pol, val))
                dates.append(date)
        else:
            skip = True
    if not(skip):
        byday = []
        udate = set(dates[1:])
        
        total = 0
        for g in udate:
            count = 0.0
            counter = 0.0
            for i in data:
                if i[0]==g:
                    count+=float(i[2])
                    counter+=1.0
            avgval = count/counter
            byday.append((g, avgval))
            total += avgval
        
        byday = map(lambda x: (parse(x[0]), x[1]), byday)
        
        sortedday = sorted(byday, key=lambda x: x[0])        
        avg_set = total / len(byday)
        
        readahead = iter(sortedday)
        next(readahead)
        
        newday = []
        for a, b in itertools.izip(sortedday, readahead):
            newday.append((b[0], b[1] - a[1]))
        
        newday = map(lambda x: (x[0], 100*x[1]/avg_set), newday)
        
        weatherday = []
        
        for i in newday:
            year = str(i[0].year)
            month = str(i[0].month)
            day = str(i[0].day)
            if len(month) < 2:
                month = '0'+month
            if len(day) <2:
                day = '0'+day
            r = requests.get('http://api.wunderground.com/api/09fc0bc8e6d8b884/history_'+year+month+day+'/q/BR/Sao_paulo.json')
            out = r.json()['history']['dailysummary'][0]
            #output: date, %change, tempm, maxhum, minhum, wspdm, pressurem, dewptm
            weather = [out['meantempm'], out['maxhumidity'], out['minhumidity'], out['meanwindspdm'], out['meanpressurem'], out['meandewptm']]
        
            weatherday.append([i[0], i[1]]+weather)
            
        with open('change'+filename, 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['date', '%change', 'tempm', 'maxhum', 'minhum', 'wspdm', 'pressurem', 'dewptm'])
            for i in weatherday:
                writer.writerow(i)
        print filename+'  written'