#!/usr/bin/python

from dataparser import *
from data import *
import time
import numpy

print('Begin smalltest')
reader = DataReader('data/smalltest.tsv', test=1)
start = time.time()
reader.readInput()
end = time.time()
data = reader.getData()
words = reader.getWords()
print ('Time to read test data: ', (end-start), ' seconds')


print ('Number of words: ', len(words))
print ('Number of data instances: ', len(data))


labelcounts = [0] * 5
featurecounts = dict.fromkeys(words, 0)
numfeatures = []
for instance in data:
    label = instance.getLabel().getLabel()
    labelcounts[label] += 1                     #count number of instances per label
    feature = instance.getFeature()
    numfeatures.append(len(feature))
    for word in instance.getWords():
        featurecounts[word] += feature.get(word)  #count number of non-zero appearances per feature

ave = numpy.average(numfeatures)
print('Average number of features per word: ', ave)
featurecountlist = []
for value in featurecounts.values():
    featurecountlist.append(int(value))
    ave = numpy.average(featurecountlist)
print('Average number of appeareances per feature: ', ave)




