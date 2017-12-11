import csv

y = []

subFile = open('subtitles.csv', 'rb')
dFile = open('dialogues.csv', 'rb')
subtitles = csv.reader(subFile, delimiter=',',quotechar='"') #Assuming this isn't very long
dialogues = csv.reader(dFile, delimiter=',' ,quotechar='"')


y = [0] * sum(1 for row in subtitles)
subFile.seek(0)

MAX_EDIT_DISTANCE = 5

#Expensive Operation
def editDistance(mat,i,j,w1,w2):
	if(mat[i][j] == -1 ):
		if i==0:
			mat[i][j] = j
		elif j==0:
			mat[i][j] = i
		else:
			mat[i][j] = min(editDistance(mat,i-1,j,w1,w2)+1,editDistance(mat,i,j-1,w1,w2)+1, (editDistance(mat,i-1,j-1,w1,w2) if w1[i]==w2[j] else editDistance(mat,i-1,j-1,w1,w2)+2 ) )
	return mat[i][j]

#Function used to simplify line so that it's comparable
def cleanLine(line):
	line.replace("n'","ng") #This is Workin' 
	return line.lower()

def isMatching(line1, line2):
	l1 = cleanLine(line1)
	l2 = cleanLine(line2)
	#mat = [[-1 for x in range(len(l2)+1)] for x in range(len(l1)+1)] 

	if ((l1 in l2) or (l2 in l1)): # or (editDistance(mat,len(l1),len(l2),"#"+l1,"#"+l2) <= MAX_EDIT_DISTANCE ) ):
		return 1
	return 0

def setSubtitleIndex(dialogue,subtitles):
	for i,line in enumerate(subtitles):
		if(isMatching(line[3], dialogue)):
			y[i] = 1
	subFile.seek(0)
	return

for row in dialogues:
	setSubtitleIndex(row[0],subtitles)


with open('__featureSet.csv', 'wb') as _featureFile:
 
	spamwriter = csv.writer(_featureFile, delimiter=',',
	                            quoting=csv.QUOTE_MINIMAL)		
	subFile.seek(0)
	oneFlag = False
	temp = []
	for i,line in enumerate(subtitles):
		#Merge LINES
		#First occurence of 1
		if(y[i] == 1 and oneFlag == False):
			temp = line
			oneFlag = True
		elif(y[i] == 1):
			temp[2] = line[2] #Change previous time to current ending time
			temp[3] += line[3]
		else:
			if(oneFlag==True):
				oneFlag = False
				spamwriter.writerow(temp + [1] )
			spamwriter.writerow(line + [y[i]] )

subFile.close()
dFile.close()