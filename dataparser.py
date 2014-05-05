#!/usr/bin/python

import csv
from data import*

class DataReader:

    def __init__(self, f, opt=0, test=0):
        self.inputfile = f
        self.datainstances = []     #list of instances (label, feature)
        self.words = []             #list of all words
        self.opt = opt              #0: binary, 1: count
        self.test = test            #1: reading in test data, ignore label
        
    def readInput(self):
        with open(self.inputfile, 'r') as inputfile:
            reader = csv.reader(inputfile, delimiter='\t')
            for row in reader:
                if self.test==0:
                    label = Label(row[3])
                else:
                    label = Label(-1)
                phrase = row[2]
                feature = self.createFeature(phrase)
                instance = Instance(label, feature)
                self.datainstances.append(instance)


    def createFeature(self, phrase):
        words = phrase.split()

        feature = Feature()
        for word in words:
            if not word in self.words:
                self.words.append(word)
            if (self.opt == 0):
                feature.add(word, 1)
            else:
                if word in feature:
                    count = feature.get(word)
                    feature.add(word, count+1)
                else:
                    feature.add(word, 1)
        return feature

    def getData(self):
        return self.datainstances

    def getWords(self):
        return self.words
            
        


