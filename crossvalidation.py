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
            #mysvm = SVM(GradientDescent())
            #mysvm = SVM(NewtonApproximation(RBF(sigma=6,caching = 1),huberparam=.01))
            mysvm = MulticlassSVM(StochasticSubgradient(param = 0.001, sample_portion = 8, iterations = 200))
            #mysvm = MulticlassSVM(GradientDescent())
            self.formSets()
            print ('Number of test samples:', len(self.test))
            self.runtraining(mysvm)
            #self.multiclass_evaluate(mysvm)
            self.direction_evaluate(mysvm)            
            end = time.time()

            print ('Total time for round:', (end - start) / 60 , 'minutes')

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
