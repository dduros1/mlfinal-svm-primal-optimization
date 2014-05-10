#!/usr/bin/python


from optimizer import *
from data import *

#############################################
#                                           #
#   This class implements a linear SVM      #
#                                           #
#############################################
class SVM:

    weights = Feature()

    def __init__(self, optimizer):
        self.optimizer = optimizer



    #############################################
    #                                           #
    #   given an instance, classify it          #
    #                                           #
    #############################################
    def predict(self, instance):
        if self.sign(instance) > 0:
            return Label(1)
        else:
            return Label(0)

    #############################################
    #                                           #
    #   given a list of training instances      #
    #                                           #
    #############################################
    def train(self, instances):
        self.weights = self.optimizer.train(instances)


    #############################################
    #                                           #
    #   gives which side of the hyper plane     #
    #     an instance should be on              #
    #                                           #
    #   TODO extend to multiclass hyperplane    #
    #                                           #
    #############################################
    def sign(self, instance):
        if instance.getFeature().dot(self.weights) >= 0:
            return 1
        return -1


#class LaplacianSVM:


class MulticlassSVM(SVM):
    

    def __init__(self, optimizer):
        super(MulticlassSVM, self).__init__(optimizer)
        self.weights = []       ##List of features


    def predict(self, instance):
        signvals = []
        for weight in self.weights():
            signvals.append(self.sign(instance)
        print instance.getLabel(), signvals

    def train(self, instances):

##################################### TESTING #########################################

def main():
    reader = DataReader('data/train.tsv', punct=1, binary=1)
    reader.readInput()
    data = reader.getData()
    print('Data Read :)')
    
    
    print ('Average accuracy:', tester.average())

if __name__ == "__main__":
    main()

    

