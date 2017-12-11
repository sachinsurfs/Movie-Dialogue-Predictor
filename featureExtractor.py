import csv
import abc
from collections import Counter


subFile = open('_featureSet.csv', 'rb')
featureFile = open('featureSet.csv', 'wb')

_features = csv.reader(subFile, delimiter=',',quotechar='"') #Assuming this isn't very long

lineDict = {}
for row in _features:
	if row[3] not in lineDict:
		lineDict[row[3]] = 0
	lineDict[row[3]] += 1

subFile.seek(0)


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
		if any((c in chars) for c in x_list[3]):			
			return 1
		return 0

class SentenceFrequency(FeatureParser):
	
	def vectorize(self, x_list): 
		if lineDict[x_list[3]] > 2:
			if(len(x_list[3].strip().split())>1):
				print x_list[3]		
				return 1
		return 0


classifier = [EorQ,SentenceFrequency]


featureWriter = csv.writer(featureFile, delimiter=',',
                            quoting=csv.QUOTE_MINIMAL)		

test11 = []
for row in _features:
	test11.append(row[-1])
print test11

for row in _features:
	feature = []
	for c in classifier:
		feature.append(c().vectorize(row[:-1]))
	featureWriter.writerow(feature + row[-1:])
subFile.close()
featureFile.close()