#!/usr/local/bin/python3

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


##################################### TESTING #########################################

#mysvm = SVM(GradientDescent())


    

