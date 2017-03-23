import sys
import csv
from dateutil.parser import parse

# def writeline(x)


def main():
	reader1 = csv.reader(open(airpollution))

	reader2 = csv.reader(open(weather1))
	
	reader2.next()
	reader1.next()
	weather = {}
	for read in reader2:
		x =  read[0].split(" on")
		x = "".join(x)
		x = str(parse(x)).split(":")
		# print x[0]
		weather[x[0]] = read
		# if "2016-04-12" in x[0]:
		# 	print x[0]
	i = 0
	for read in reader1:
		x = read[4].split("T")
		x = " ".join(x)
		x = str(parse(x)).split(":")
		fd = open(cityname+read[5]+'.csv','a')
		fd = csv.writer(fd)
		if x[0] not in weather:
			continue
		w = weather[x[0]]
		fd.writerow([read[0],w[0],read[5],read[6],read[7],w[5],w[7],w[8],w[9],w[11],w[15],w[18],w[13]])
		
		i =1

if __name__=='__main__':
	airpollution = sys.argv[1]
	weather1 = sys.argv[2]
	cityname = sys.argv[3]
	main()