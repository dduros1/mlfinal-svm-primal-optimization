#!/usr/bin/python

import copy
from data import *

class Optimizer:

    def __init__(self):
        self.stuff = stuff




#############################################
#                                           #
#       Gradient Descent Optimizer          #
#                                           #
#############################################
class GradientDescent(Optimizer):

    weights = Feature()

    #TODO: make learning rate variable
    def __init__(self, iterations=10, learning_rate=0.001):
        self.iterations = iterations
        self.learning_rate = learning_rate

    def train(self, instances):
        for i in xrange(0, self.iterations):
            for instance in instances:
                x = copy.copy(instance.getFeature())
                print type(x)
                print type(instance.getFeature())
                print type(instance)
                f = instance.getFeature()
                print type(f)
                true_label = instance.getLabel()
                if (true_label == 0):
                    true_label = -1

                if x.dot(self.weights)*true_label < 1:
                    x.scalar_multiply(self.learning_rate*true_label)
                    self.weights.plusall(x)
        return self.weights


#############################################
#                                           #
#       Stochastic Subgradient optimizer    #
#                                           #
#############################################
class StochasticSubgradient(Optimizer):

    def __init__(self):
        self.stuff = stuff


#############################################
#                                           #
#       Newton Primal Optimizer             #
#                                           #
#############################################
class NewtonApproximation(Optimizer):

    def __init__(self):
        self.stuff = stuff



