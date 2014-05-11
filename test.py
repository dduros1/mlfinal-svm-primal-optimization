#!/usr/bin/python


from dataparser import *
from data import *
import time
import numpy
import operator

#print('Begin smalltest')
print ('running on training data')
reader = DataReader('data/train.tsv', test = 1, punct = 1, lower = 1)
#reader = DataReader('data/train.tsv', punct=1)
start = time.time()
reader.readInput()
end = time.time()
data = reader.getData()
words = reader.getWords()
print ('Time to read data: ', (end-start), ' seconds')


print ('Number of words: ', len(words))
print ('Number of data instances: ', len(data))


labelcounts = [0] * 5
featurecounts = dict.fromkeys(words, 0)
numfeatures = []
for instance in data:
    label = instance.getLabel().getLabel()
    labelcounts[int(label)] += 1                     #count number of instances per label
    feature = instance.getFeature()
    numfeatures.append(len(feature))
    for word in instance.getWords():
        featurecounts[word] += feature.get(word)  #count number of non-zero appearances per feature

ave = numpy.mean(numfeatures)
#print('Average number of features per word: ', ave)
featurecountlist = []
for value in featurecounts.values():
    featurecountlist.append(int(value))
ave = numpy.average(featurecountlist)
print('Average number of appeareances per feature: ', ave)

#sorted_features = sorted(featurecounts.iteritems(), key=operator.itemgetter(1))

#print sorted_features[:10]
#print ('10 most common words')
#print sorted_features[len(sorted_features)-10:]

#num_one = 0
#for (word, count) in sorted_features:
#    if count==1:
#        num_one += 1
#print ('number of words that appear once', num_one)


