# -*- coding: utf-8 -*-
"""
Created on Sat May 07 17:30:36 2016

@author: ericy
"""

import requests

pol = ['pm25', 'pm10', 'so2', 'no2', 'o3', 'co', 'bc']

#cities = requests.get('https://api.openaq.org/v1/cities', params={'limit':'1000'}).json()

for i in pol:
    filename = i+'SP.csv'
    payload = {'parameter':i, 'format':'csv', 'country':'BR', 'city':'São Paulo'}

    with open(filename, 'wb') as outfile:
        #outfile.write(jout.encode('UTF-8'))

        r_csv = requests.get("https://api.openaq.org/v1/measurements", params=payload)
        #print r_csv.text
        outfile.write(r_csv.text.encode('UTF-8'))