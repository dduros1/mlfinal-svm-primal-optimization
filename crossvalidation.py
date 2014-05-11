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

    def __init__(self, instances, optimizer, multi, rounds = 10):
        self.alldata = instances
        self.optimizer = optimizer
        self.multi = multi
        self.rounds

    def runtest(self):
        for round in range(self.rounds):
            if self.multi == 0:
                mysvm = SVM(self.optimizer)
            else:
                mysvm = MulticlassSVM(self.optimizer)
            self.formSets()
            self.runtraining(mysvm)
            self.evaluate(mysvm)            
          
    def formSets(self):
        datalist =  copy.copy(self.alldata)
        random.shuffle(datalist)
        self.test = datalist[:int(len(self.alldata) * 0.1)]
        self.train = datalist[int(len(self.alldata) * 0.1):]   

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

    def multiclass_evaluate(self, mysvm):
        dists = []
        wrongdists = []
        for instance in self.test:
            newlabel = mysvm.predict(instance)
            dist = abs(newlabel.getLabel() - instance.getLabel().getLabel())
            dists.append(dist)
            if not dist == 0:
                wrongdists.append(dist)
        aver = sum(dists) * 1.0 / len(dists)
        self.results.append(aver)
        print('Average distance:', aver)
        aver = sum(wrongdists) * 1.0 / len(wrongdists)
        print('Average wrong distance:', aver)        

    def direction_evaluate(self, mysvm):
        equal = over = under = 0        
        for instance in self.test:
            newlabel = mysvm.predict(instance)
            direc = newlabel.getLabel() - instance.getLabel().getLabel()
            if direc == 0:
                equal += 1
            elif direc > 0:
                over += 1
            else:
                under += 1
        self.results.append(equal * 1.0 / len(self.test))
        print('Equal:', equal * 1.0 / len(self.test))
        print('Overestimate:', over * 1.0 / len(self.test))
        print('Underestimate:', under * 1.0 / len(self.test))
        
    def average(self):
        total = sum(self.results)
        ave = total/len(self.results)
        return ave
        


###########################  TEST  #########################################

def main():
    reader = DataReader('data/train.tsv', punct=1, binary=1, lower = 1)
    reader.readInput()
    data = reader.getData()
    print('Data Read :)')
    tester = CrossValidationTester(data)
    tester.runtest()
    print ('Average accuracy:', tester.average())

if __name__ == "__main__":
    main()
