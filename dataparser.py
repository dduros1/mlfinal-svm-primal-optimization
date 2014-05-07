#!/usr/bin/python

import csv
from data import*
import string

#############################################
#                                           #
#       DataReader for SVM class            #
#                                           #
#############################################
class DataReader:

    punctuation = [',','.','?','!','\'s','/']    #Punctuation vector for removal purposes

    #############################################
    #                                           #
    #       Only req. param: input file         #
    #                                           #
    #############################################
    def __init__(self, f, opt=0, test=0, punct=0, binary=0):
        self.inputfile = f
        self.datainstances = []     #list of instances (label, feature)
        self.words = []             #list of all words
        self.opt = opt              #0: binary, 1: count
        self.test = test            #1: reading in test data, ignore label
        self.punct = punct          #1: ignore punctuation
        self.binary = binary        #1: don't collapse ratings to 0/1
        
    #############################################
    #                                           #
    #       Reads input from provided file      #
    #                                           #
    #############################################
    def readInput(self):
        with open(self.inputfile, 'r') as inputfile:
            reader = csv.reader(inputfile, delimiter='\t')
            count = 0
            first = True
            for row in reader:
                if self.test==0:
                    if first:
                        first = False
                        continue
                    label = self.createLabel(row[3])#Label(row[3])
                    
                else:
                    label = Label(-1)
                phrase = row[2]
                feature = self.createFeature(phrase)
                if len(feature) > 0:
                    instance = Instance(label, feature)
                    self.datainstances.append(instance)

    #############################################
    #                                           #
    #       Create label for training data      #
    #                                           #
    #############################################
    def createLabel(self, label):
        if self.binary == 1:
            return Label(label)
        if (label > 2):
            return Label(1)
        return Label(0)

    #############################################
    #                                           #
    #       Create feature function             #
    #                                           #
    #############################################
    def createFeature(self, phrase):
        words = phrase.split()
        #TODO tolower()?
        feature = Feature()
        for word in words:
            if self.punct == 1:
                if word in self.punctuation:
                    continue
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

    #############################################
    #                                           #
    #       Returns all data instances read     #
    #                                           #
    #############################################
    def getData(self):
        return self.datainstances

    #############################################
    #                                           #
    #   Returns list of all indiv. words read   #
    #                                           #
    #############################################
    def getWords(self):
        return self.words
            
        

