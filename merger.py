import csv
import re

y = []

subFile = open('subtitles.csv', 'rb')
dFile = open('dialogues.csv', 'rb')
subtitles = csv.reader(subFile, delimiter=',',quotechar='"') #Assuming this isn't very long
dialogues = csv.reader(dFile, delimiter=',' ,quotechar='"')


y = [0] * sum(1 for row in subtitles)
subFile.seek(0)

MAX_EDIT_DISTANCE = 5


contractions = { 
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

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
	line = line.replace("n'","ng") #This is Workin'
	linelist = line.split()
	for w in linelist:
		if w in contractions:
			line = line.replace(w,contractions[w])

	line = re.sub('[,\'!@#$%^&*()+-=]', '', line)
	return line.lower()

#Line 1 - subtitle Line 2 - dialouge
def isMatching(line1, line2):
	if (len(line1.split())==1) : #probably not a dialogue
		return 0

	l1 = cleanLine(line1)
	l2 = cleanLine(line2)
	#mat = [[-1 for x in range(len(l2)+1)] for x in range(len(l1)+1)] 
	
	if (l1 in l2) or (l2 in l1): # or (editDistance(mat,len(l1),len(l2),"#"+l1,"#"+l2) <= MAX_EDIT_DISTANCE ) ):
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