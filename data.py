#!/usr/bin/python

import math

#############################################
#                                           #
#           Data instance class             #
#                                           #
#############################################
class Instance:

    #############################################
    #                                           #
    #   Instances require a label and a feature #
    #                                           #
    #############################################
    def __init__(self, label, feature):
        self.label = label
        self.feature = feature

    #############################################
    #                                           #
    #   Returns label associated with instance  #
    #                                           #
    #############################################
    def getLabel(self):
        return self.label

    def getID(self):
        return self.feature.getID()

    #############################################
    #                                           #
    #  Returns feature associated with instance #
    #                                           #
    #############################################
    def getFeature(self):
        return self.feature

    #############################################
    #                                           #
    #   Implement equals for instances          #
    #                                           #
    #############################################
    def equals(self, i2):
        if self.label.equals(i2.getLabel) and self.feature.equals(i2.getFeature):
            return True
        return False
            
    #############################################
    #                                           #
    # Returns words associated with instance's  #
    #   feature                                 #
    #                                           #
    #############################################
    def getWords(self):
        return self.feature.getWords()

    #############################################
    #                                           #
    #           Override string method          #
    #                                           #
    #############################################

    def __str__(self):
        return (self.label.__str__(), self.feature.__str__()).__str__()

    def __repr__(self):
        return str(self)





#############################################
#                                           #
#                Label class                #
#                                           #
#############################################
class Label:
    
    #############################################
    #                                           #
    #          Requires integer label           #
    #                                           #
    #############################################
    def __init__(self, label):
        self.label = label;
    
    #############################################
    #                                           #
    #           Returns label value             #
    #                                           #
    #############################################
    def getLabel(self):
        return self.label

    #############################################
    #                                           #
    #       Implement equals for Label class    #
    #                                           #
    #############################################
    def equals(self, l2):
        if not isinstance(l2, Label):
            return False
        if (self.getLabel() == l2.getLabel()):
            return True
        return False

    #############################################
    #                                           #
    #           Override string method          #
    #                                           #
    #############################################

    def __str__(self):
        return str(self.label)

    def __repr__(self):
        return str(self)

#############################################
#                                           #
#                Feature Class              #
#                                           #
#############################################
class Feature:
    
    def __init__(self, uniqueid = -1):
        self.features = {}
        self.uniqueid = uniqueid

    #############################################
    #                                           #
    #   Get feature that corresp to word i      #
    #                                           #
    #############################################
    def get(self, word):                         
        if (word in self.features):
            return self.features[word]
        else:
            return 0

    def getID(self):
        return self.uniqueid

    #############################################
    #                                           #
    #    Adds feature associated with word i    #
    #                                           #
    #############################################
    def add(self, word, value):
        if value != 0:
            self.features[word] = value
        else:
            if word in self.features:
                del self.features[word]
    #############################################
    #                                           #
    #       Override contains method            #
    #                                           #
    #############################################
    def __contains__(self, word):
        if word in self.features:
            return True
        return False

    #############################################
    #                                           #
    #       Return words in feature             #
    #                                           #
    #############################################
    def getWords(self):
        return self.features.keys()

    #############################################
    #                                           #
    #           Override length                 #
    #                                           #
    #############################################
    def __len__(self):
        return len(self.features)

    #############################################
    #                                           #
    #            Dot product                    #
    #                                           #
    #############################################
    def dot(self, f2):
        sumval = 0
        for word in self.getWords():
            sumval += self.get(word)*f2.get(word)
        return sumval

    #############################################
    #                                           #
    #   Plus gets for all words                 #
    #                                           #
    #############################################

    def plusall(self, f2):
        for word in f2.getWords():
            self.add(word, f2.get(word)+self.get(word))

    #############################################
    #                                           #
    #            Scalar product                 #
    #                                           #
    #############################################
    def scalar_multiply(self, val):
        for word in self.getWords():
            self.add(word, val*self.get(word))

    #############################################
    #                                           #
    #           Override string method          #
    #                                           #
    #############################################

    def __str__(self):
        return str(self.features)

    def __repr__(self):
        return str(self)


    #############################################
    #                                           #
    #        Norm method for RBF kernel         #
    #                                           #
    #############################################
    def squared_norm(self, f2):
        norm = Feature()
        allwords = self.getWords() + f2.getWords()

        diffs = []
        for word in allwords:
            diff = self.get(word) - f2.get(word)
            diff = diff * diff
            diffs.append(diff)
        return sum(diffs)

    def self_norm(self):
        return self.dot(self)        
        #val = 0        
        #for word in self.getWords():
        #    val += self.get(word) * self.get(word)
        #return math.sqrt(val)

    def sum(self):
        val = 0
        for word in self.features:
            val += self.get(word)
        return val

