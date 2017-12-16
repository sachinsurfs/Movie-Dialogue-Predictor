import csv
import sys


if len(sys.argv)==1:
	movie_name = "Ant-Man"
else:
	movie_name = sys.argv[1]

file_name = "datasets/srt/"+movie_name+"-subs.srt"
file=open(file_name,"r+")

#TODO :Consume srt last line

csvfile= open("datasets/subtitles/"+movie_name+"_subtitles.csv","wb")
csvout= csv.writer(csvfile)

arr=[]


line = "-"
while line:
	index = file.readline().strip()

	time_start,time_end = file.readline().strip().split(" --> ")

	dialogue = file.readline().strip().split("- ")[::-1][0]

	line = file.readline()
	while line.strip():

		monologue = line.strip().split("- ")
		if(len(monologue)==1):
			dialogue += ' '+monologue[0]
		else: #New Character introduced
			csvout.writerow([index,time_start,time_end,dialogue])
			dialogue = monologue[1]

		line = file.readline()
	csvout.writerow([index,time_start,time_end,dialogue])

file.close()
csvfile.close()