#!/usr/bin/python

import copy
from data import *
import time
from dataparser import *


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
    def __init__(self, iterations = 10, learning_rate = 0.001):
        self.iterations = iterations
        self.learning_rate = learning_rate

    def train(self, instances):
        for i in range(0, self.iterations):
            for instance in instances:
                x = copy.copy(instance.getFeature())
                f = instance.getFeature()
                true_label = instance.getLabel().getLabel()
                if (true_label == 0):
                    true_label = -1

                if x.dot(self.weights) * true_label < 1:
                    x.scalar_multiply(self.learning_rate * true_label)
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

    weights = Feature()

    def __init__(self, kernel,huberparam=0.01):
        self.huberparam = huberparam
        self.kernel = kernel


    def train(self, instances):
        start = time.time()
        kernelinstances = self.compute_kernel(instances)
        end = time.time()
        print ('Kernel computed in', (end-start)/60, 'seconds')
        beta = self.primalsvm(instances, kernelinstances)
        #weights = sum beta * inst
        
        
    def primalsvm(self, instances, kernelinstances):
        num = len(instances)
        sv = []
        if num > 1000:
            small_instances = instances[:num/2]
            beta = self.primalsvm(small_instances, kernelinstances)
            for key, value in beta:
                if not value == 0:
                    sv.append(key)
        else:
            sv = copy.copy(instances)
        oldsv = copy.copy(sv)
        while (oldsv == sv):
            #beta = invert(k_sv + lambda on the diagonals) * labels_sv
            pass
        return beta


    def compute_kernel(self, instances):
        kernels ={}
        for instance in instances:
            instdict = {}
            for instance2 in instances:
                if not instance.equals(instance2):
                    kval = self.kernel.K(instance.getFeature(), instance2.getFeature())
                    instdict[instance2] = kval
            kernels[instance] = instdict

    def form_matrix(self, kernelinstances, sv):
        from numpy import matrix
        from scipy import linalg

        n = len(kernelinstances)
        np.empty([n,n]) 




#############################################
#                                           #
#      Kernel Class                         #
#                                           #
#############################################
class Kernel:
    pass



#############################################
#                                           #
#           RBF kernel                      #
#                                           #
#############################################
class RBF(Kernel):
    
    def __init__(self, sigma=2):
        self.sigma = sigma

    #x1, x2 are Features
    def K(self, x1, x2):
        import math
        val = math.exp(-1*x1.squared_norm(x2)/(2*self.sigma*self.sigma))
        return val


#############################################
def main():
    reader = DataReader('data/smalltrain.tsv', punct=1)
    reader.readInput()
    data = reader.getData()
    print('Data Read :)')
    tester = NewtonApproximation(RBF())
    tester.train(data)

if __name__ == "__main__":
    main()

