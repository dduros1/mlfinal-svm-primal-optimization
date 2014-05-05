#!/usr/bin/python

from dataparser import *
from data import *
import time
import numpy

reader = DataReader('data/test.tsv', test=1)
start = time.time()
reader.readInput()
end = time.time()
data = reader.getData()
words = reader.getWords()
print 'Time to read train data: ', (end-start), ' seconds'


print 'Number of words: ', len(words)
print 'Number of data instances: ', len(data)


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

ave = np.average(numfeatures)
ave = np.average(featurecounts.getValues())





