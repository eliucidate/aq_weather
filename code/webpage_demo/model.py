import pico
import urllib2
import json
from sklearn.ensemble import RandomForestRegressor
import csv

path = 'http://api.wunderground.com/api/972ef893cd0f7a2d/'
weather_dict = {"cloudy": 0.0, "mostlycloudy": 1.0, "rain": -1.0, "snow": -2.0, "tstorms": -3.0, "fog": 0.0, "partlycloudy": 2.0, "clear": 3.0}
invert_weather_dict = {0.0: "Cloudy", 1.0: "Mostly cloudy", -1.0: "Rain", -2.0: "Snow", -3.0: "Tstorms", 2.0: "Partly cloudy", 3.0: "Clear"}
city_dict = {"New York": "NY/New_York", "San Francisco": "CA/San_Francisco", "Houston": "TX/Houston", "Boston": "MA/Boston"}

def predict_1(city, weather, temp, wind_speed, humi):
	if city == "Choose your city":
		city = ""
	weather = weather_dict[weather]
	temp = float(temp)
	wind_speed = float(wind_speed)
	humi = float(humi)
	weather_params = [weather, temp, humi, wind_speed]
	# model
	# Read Training data
	csv_reader = csv.reader(open('data/predict/NO2.txt', 'rU'), delimiter='\t')#delimiter="\t"
	data_no2 = [x for x in csv_reader]
	csv_reader = csv.reader(open('data/predict/co.txt', 'rU'), delimiter='\t')
	data_co = [x for x in csv_reader]
	csv_reader = csv.reader(open('data/predict/pm25.txt', 'rU'), delimiter='\t')
	data_pm25 = [x for x in csv_reader]
	prediction = []
	for pollutant_set in [data_no2,data_co,data_pm25]:
		target = [x[0] for x in pollutant_set]
		train = [x[1:5] for x in pollutant_set]
		#create and train the random forest
		#multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
		rg = RandomForestRegressor(n_estimators=100, n_jobs=4)
		rg.fit(train, target)
		#Return Prediction
		prediction.extend(rg.predict(weather_params))
	return city + "</br>PM25: &nbsp;&nbsp;" + str("{0:.3f}".format(prediction[2])) + " ug/m3</br>CO: &nbsp;&nbsp;" + str("{0:.3f}".format(prediction[1])) + " ppm</br>NO2: &nbsp;&nbsp;" + str("{0:.3f}".format(prediction[0])) + " ppm"

def predict_2(city):
	if city == "Choose your city":
		return "Please select a city."
	city = city_dict[city]
	weather_parameters = get_weather(city)

	#Read Training data
	csv_reader = csv.reader(open('data/predict/NO2.txt', 'rU'), delimiter='\t')#delimiter="\t"
	data_no2 = [x for x in csv_reader]
	csv_reader = csv.reader(open('data/predict/co.txt', 'rU'), delimiter='\t')
	data_co = [x for x in csv_reader]
	csv_reader = csv.reader(open('data/predict/pm25.txt', 'rU'), delimiter='\t')
	data_pm25 = [x for x in csv_reader]
	predictions = [[] for i in range(len(weather_parameters))]
	for pollutant_set in [data_no2,data_co,data_pm25]:
		target = [x[0] for x in pollutant_set]
		train = [x[1:5] for x in pollutant_set]
		#create and train the random forest
		#multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
		rg = RandomForestRegressor(n_estimators=100, n_jobs=4)
		rg.fit(train, target)
		#Return Prediction
		for i in range(len(weather_parameters)):
			if weather_parameters[i][1] in weather_dict:
				weather_parameters[i][1] = weather_dict[weather_parameters[i][1]]
			else:
				weather_parameters[i][1] = 0.0
			pred = rg.predict(weather_parameters[i][1:])
			predictions[i].extend(pred) #TO BE PREDICTED
	res = city + "</br></br>"
	for i in range(len(weather_parameters)):
		res += weather_parameters[i][0] + ":</br>"
		res += "weather: " + invert_weather_dict[weather_parameters[i][1]] + ";&nbsp;&nbsp; temp: " + str(weather_parameters[i][2]) + ";&nbsp;&nbsp; humidity: " + str(weather_parameters[i][3]) + ";&nbsp;&nbsp; wind speed: " + str(weather_parameters[i][4]) + "</br>"
		res += "AQ:&nbsp;&nbsp; PM25: " + str("{0:.3f}".format(predictions[i][2])) + ";&nbsp;&nbsp; CO: " + str("{0:.3f}".format(predictions[i][1])) + ";&nbsp;&nbsp; NO2: " + str("{0:.3f}".format(predictions[i][0])) + "</br></br>"
	return res

def get_weather(city):
	f = urllib2.urlopen(path+'forecast/q/'+city+'.json')
	json_string = f.read()
   	parsed_json = json.loads(json_string)
   	forecast = parsed_json["forecast"]["simpleforecast"]
   	observations = forecast["forecastday"]
   	multi_day_weather = []
   	for ob in observations:
   		date = ob["date"]
   		year = date["year"]
		mon = date["monthname_short"]
		day = date["day"]
		weekday = date["weekday"]
		weather = ob["icon"]
		tempH = ob["high"]["fahrenheit"]
		tempL = ob["low"]["fahrenheit"]
		maxwind = ob["maxwind"]["mph"]
		avewind = ob["avewind"]["mph"]
		humi = ob["avehumidity"]
		date = mon + " " + str(day) + ", " + str(year)
		weather_params = [date, str(weather), float(tempH), float(humi), float(avewind)]
		multi_day_weather.append(weather_params)
	return multi_day_weather   	 













