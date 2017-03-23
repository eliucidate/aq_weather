import urllib2
import json
import csv
from datetime import timedelta, date
import sys
# This is the api to get historical weather qualities
path = 'http://api.wunderground.com/api/972ef893cd0f7a2d/'

def daterange(start_date, end_date):
   for n in range(int ((end_date - start_date).days)):
       yield start_date + timedelta(n)

def getdata(start_date, end_date, zipcode):
	writer = csv.writer(open(zipcode+"weather.csv","w"))
	writer.writerow(["date", "mon", "mday", "hour","year","weather","tempm","tempi","hum","wspdm","wspdi","wgustm","wgusti","vism","visi","pressurem","pressurei","windchillm","windchilli","heatindexm","heatindexi","conds"])
	for single_date in daterange(start_date, end_date):
   		datename = single_date.strftime("%Y%m%d")
   		f = urllib2.urlopen(path+'history_'+datename+'/q/'+zipcode+'.json')
   		json_string = f.read()
   		parsed_json = json.loads(json_string)
   		# print parsed_json
   		history = parsed_json["history"]
		observations = history["observations"]
		for ob in observations:
			date = ob["date"]
			pretty = date["pretty"]
			year = date["year"]
			mon = date["mon"]
			mday = date["mday"]
			hour = date["hour"]
			time = mon+' '+mday+' '+hour+' '+year
			weather = ob["icon"]
			tempm = ob["tempm"]
			tempi = ob["tempi"]
			hum = ob["hum"]
			wspdm = ob["wspdm"]
			wspdi = ob["wspdi"]
			wgustm = ob["wgustm"]
			wgusti = ob["wgusti"]
			vism = ob["vism"]
			visi = ob["visi"]
			pressurem = ob["pressurem"]
			pressurei = ob["pressurei"]
			windchillm = ob["windchillm"]
			windchilli = ob["windchilli"]
			heatindexi = ob["heatindexi"]
			heatindexm = ob["heatindexm"]
			conds = ob["conds"]
			writer.writerow([pretty,mon,mday,hour,year,weather,tempm,tempi,hum,wspdm,wspdi,wgustm,wgusti,vism,visi,pressurem,pressurei,windchillm,windchilli,heatindexm,heatindexi,conds])
			# print time,weather,tempm,tempi
def main():
	# start_date = date(2016,3,6)
	# end_date = date(2016,5,12)

	start_date = date(year1,month1,day1)
	end_date = date(year2,month2,day2)
	
	getdata(start_date,end_date,place)
if __name__=='__main__':
	if len(sys.argv) < 3:
		print "error: parameters are not enough"
	year1 = int(sys.argv[1])
	month1 = int(sys.argv[2])
	day1 = int(sys.argv[3])
	year2 = int(sys.argv[4])
	month2 = int(sys.argv[5])
	day2 = int(sys.argv[6])
	place = sys.argv[7]

	main()

# weather = parsed_json["observations"]["icon"]
# # temp_f  =  parsed_json["current_observation"]["temp_f"]
# print str(weather)


	