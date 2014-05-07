#!/usr/local/bin/python3


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
        if (self.label == l2.getLabel):
            return True
        return False

    #############################################
    #                                           #
    #           Override string method          #
    #                                           #
    #############################################

    def __str__(self):
        return self.label

#############################################
#                                           #
#                Feature Class              #
#                                           #
#############################################
class Feature:
    
    def __init__(self):
        self.features = {}

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
                self.features.remove(word)
    #############################################
    #                                           #
    #       Override contains method            #
    #                                           #
    #############################################
    def __contains__(self, word):
        if word in self.features:
            return true
        return false

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
        for word in self.features:
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
        for word in self.features:
            self.add(word, val*self.get(word))

    #############################################
    #                                           #
    #           Override string method          #
    #                                           #
    #############################################

    def __str__(self):
        return self.features



