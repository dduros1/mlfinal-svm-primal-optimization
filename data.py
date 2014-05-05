#!/usr/bin/python

""" Data instance class  """
class Instance:

    def __init__(self, label, feature):
        self.label = label
        self.feature = feature


    def getLabel(self):
        return self.label

    def getFeature(self):
        return self.feature

    def equals(self, i2):
        if self.label.equals(i2.getLabel) and self.feature.equals(i2.getFeature):
            return True
        return False
            




""" Label class  """
class Label:
    
    def __init__(self, label):
        self.label = label;

    def getLabel(self):
        return self.label

    def equals(self, l2):
        if (self.label == l2.getLabel):
            return True
        return False


""" Feature Class   """
class Feature:
    
    def __init__(self):
        self.features = {}

    """ Get feature that corresp to word i """
    def get(self, word):                         
        if (word in self.features):
            return self.features[word]
        else:
            return 0

    """ Adds feature associated with word i """
    def add(self, word, value):
        if value != 0:
            self.features[word] = value
        else:
            if word in self.features:
                self.features.remove(word)

    def __contains__(self, word):
        if word in self.features:
            return true
        return false

    def getWords(self):
        return features.getKeys()

    def __len__(self):
        return len(self.features)




