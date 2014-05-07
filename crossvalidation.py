#!/usr/local/bin/python3


from svm import *
from optimizer import *
import random, copy
from dataparser import *
from data import *
import time
import numpy
import operator


class CrossValidationTester:

    test = []
    train = []

    def __init__(self, instances):
        self.alldata = instances

    def runtest(self):
        for round in xrange(0, 10):
            print ('Round', round)
            start = time.time()
            mysvm = SVM(GradientDescent())
            self.formSets()
            print ('Number of test samples:', len(self.test))
            self.trainshit(mysvm)
            self.evaluate(mysvm)
            end = time.time()

            print ('Total time for round:', (end-start)/60 , 'minutes')

    def formSets(self):
        self.test = random.sample(self.alldata, int(len(self.alldata)*.1))
        self.train = copy.copy(self.alldata)
        for ele in self.test:
            self.train.remove(ele)
   
    def trainshit(self, mysvm):
        mysvm.train(self.train)

    def evaluate(self, mysvm):
        correct = 0
        for instance in self.test:
            newlabel = mysvm.predict(instance)
            if newlabel.equals(instance.getLabel()):
                correct += 1

        print ('Correct:', correct/len(self.test))


###########################  TEST  #########################################


reader = DataReader('data/smalltrain.tsv', punct=1)
reader.readInput()
data = reader.getData()
print(type(data[0]))
print(data[0])
print 'Data Read :)'
tester = CrossValidationTester(data)
tester.runtest()


