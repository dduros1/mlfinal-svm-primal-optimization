#!/usr/bin/python


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
    results = []

    def __init__(self, instances):
        self.alldata = instances

    def runtest(self):
        for round in range(0, 10):
            print ('Round', round)
            start = time.time()
            mysvm = SVM(GradientDescent())
            self.formSets()
            print ('Number of test samples:', len(self.test))
            self.runtraining(mysvm)
            self.evaluate(mysvm)
            end = time.time()

            print ('Total time for round:', (end - start) / 60 , 'minutes')

    def formSets(self):
        self.test = random.sample(self.alldata, int(len(self.alldata) * 0.1))
        self.train = copy.copy(self.alldata)
        for ele in self.test:
            self.train.remove(ele)
   
    def runtraining(self, mysvm):
        mysvm.train(self.train)

    def evaluate(self, mysvm):
        correct = 0.0
        for instance in self.test:
            newlabel = mysvm.predict(instance)
            if newlabel.equals(instance.getLabel()):
                correct += 1

        self.results.append(correct/len(self.test))
        print ('Correct:', correct/len(self.test))



    def average(self):
        total = sum(self.results)
        ave = total/len(self.results)
        return ave
        


###########################  TEST  #########################################

def main():
    reader = DataReader('data/smalltrain.tsv', punct=1)
    reader.readInput()
    data = reader.getData()
    print('Data Read :)')
    tester = CrossValidationTester(data)
    tester.runtest()
    print ('Average accuracy:', tester.average())

if __name__ == "__main__":
    main()
