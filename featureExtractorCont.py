import csv
import abc
import re
from collections import Counter
from nltk.corpus import wordnet as wn
from datetime import time
import sys

if len(sys.argv)==1:
	movie_name = "Ant-Man"
else:
	movie_name = sys.argv[1]

subFile = open('datasets/_featureSets/_'+movie_name+'_featureSet.csv', 'rb')
featureFile = open('datasets/featureSets/'+movie_name+'_featureSet.csv', 'wb')

_features = csv.reader(subFile, delimiter=',',quotechar='"') #Assuming this isn't very long

lineDict = {}
wordDict = {}
for row in _features:
	if row[3] not in lineDict:
		lineDict[row[3]] = 0
	lineDict[row[3]] += 1
	line = re.sub('[?!.]', ' ', row[3]).split()
	for w in line:
		if w not in wordDict:
			wordDict[w] = 0
		wordDict[w] += 1

subFile.seek(0)

nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}

prevTime = ""

def getTime(timeString):
	timeList = timeString.split(":")
	second,milli = timeList[2].split(",")
	return time(int(timeList[0]),int(timeList[1]),int(second),int(milli)*1000)

class FeatureParser(object):
	__metaclass__ = abc.ABCMeta

	'''
	Input : list[index,time_s,time_e,line] 
	Returns: 0 or 1
	'''
	@abc.abstractmethod
	def vectorize(self, x_list): 
		pass


class EorQ(FeatureParser):
	
	def vectorize(self, x_list): 
		chars = set('!?')
		count=0
		if any((c in chars) for c in x_list[3]):			
			count+=1
		return count

class Pivot_words(FeatureParser):
	
	def vectorize(self, x_list): 
		pivot_words = ['but', 'however','inspite', 'although']
		count=0
		for word in pivot_words:
			if word in x_list[3]:			
				count+= 1
		return count
		
class Length(FeatureParser):
	
	def vectorize(self,x_list):
		return len(x_list[3])
		
class Indef_articles(FeatureParser):
	def vectorize(self, x_list): 
		indef = ['a', 'an']
		count=0
		for c in x_list[3]:
			if c in indef:
				count+=1
		return count


class SentenceFrequency(FeatureParser):
	
	def vectorize(self, x_list): 
		if(len(x_list[3].strip().split())==1):
			return 0
		return lineDict[x_list[3]]

class UncommonWord(FeatureParser):

	def vectorize(self, x_list):

		cuss = ['fuck', 'bitch', 'motherfucker', 'fucking', 'motherfucking']
		
		line = re.sub('[?!.]', ' ', x_list[3]).split()
		count=0
 		for w in line:
 			if w in nouns or cuss:
 				if wordDict[w] == 1:
 					count+= 1
		return count

class TimeGap(FeatureParser):

	def vectorize(self, x_list):
		global prevTime
		if(prevTime == ""):
			prevTime = x_list[2]
			prevTimeObject = getTime(prevTime)
			return (prevTimeObject.minute*60) + prevTimeObject.second #opening line
		
		prevTimeObject = getTime(prevTime)
		currentTimeObject = getTime(x_list[2])
		prevTime = x_list[2]

		if(prevTimeObject.minute == currentTimeObject.minute):
			gap = (currentTimeObject.second + float(currentTimeObject.microsecond)/1000000 ) - (prevTimeObject.second + float(prevTimeObject.microsecond)/1000000) 
			return gap
		if(currentTimeObject.minute - prevTimeObject.minute == 1 ):
			gap = (currentTimeObject.second + float(currentTimeObject.microsecond)/1000000 + 60) - (prevTimeObject.second + float(prevTimeObject.microsecond)/1000000) 
			return gap
		return (currentTimeObject.minute - prevTimeObject.minute)*60

class TrailingCommas(FeatureParser) :
	def vectorize(self, x_list):
		return x_list[3].count(",")

classifier = [EorQ,Pivot_words,Length,Indef_articles,SentenceFrequency,UncommonWord,TimeGap,TrailingCommas]

featureWriter = csv.writer(featureFile, delimiter=',',
                            quoting=csv.QUOTE_MINIMAL)		


for row in _features:
	feature = []
	for c in classifier:
		feature.append(c().vectorize(row[:-1]))
	featureWriter.writerow(feature + row[-1:])
subFile.close()
featureFile.close()

