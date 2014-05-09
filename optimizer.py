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

    #############################################
    #                                           #
    #      Train function: calls recursive      #
    #       function, then transforms beta      #
    #       into svm weights                    #
    #                                           #
    #############################################
    def train(self, instances):
        beta = self.primalsvm(instances)
        #weights = sum beta * inst
        for inst in instances:
            for word in inst.getWords():
                weights.add(word, weights.get(word) + beta[inst] * inst.get(word))

    #############################################
    #                                           #
    #      Recursive function for train         #
    #                                           #
    #############################################    
    def primalsvm(self, instances):
        num = len(instances)
        sv = []
        if num > 1000:
            small_instances = instances[:num/2]
            beta = self.primalsvm(small_instances)
            for key, value in beta:
                if not value == 0:
                    sv.append(key)
        else:
            sv = copy.copy(instances)
        oldsv = []
        while (not oldsv.equals(sv)):
            oldsv = copy.copy(sv)
            kmatrix = self.form_matrix(sv)
            tempmatrix = copy.copy(kmatrix)
            for i in range(n):
                tempmatrix[i][i] += self.huberparm
            inverse = tempmatrix.I
            labels = self.form_label_vec(sv)
            beta = inverse.dot(labels)
            sv = self.update(instances, beta, kmatrix, oldsv)

        return self.formdictionary(beta, sv)


    def formdictionary(self, beta, sv):
        mydict = {}
        for i in len(sv):
            mydict[sv[i]] = beta[i]
        return mydict

    #############################################
    #                                           #
    #           update sv step                  #
    #                                           #
    #############################################
    def update(self, instances, beta, kmatrix, oldsv):
        newsv = []
        for i in len(instances):
           val = 0.0
           for j in len(oldsv):
                val += self.kernel.k(instances[i].getFeature(), sv[j].getFeature()) * beta[j]
           label = instances[i].getLabel().getLabel()
           if label == 0:
               val *= -1
           if val < 1:
               newsv.append(instances[i])

        return newsv

    #############################################
    #                                           #
    #      invertible matrix calculation        #
    #       returns matrix inverse yay          #
    #                                           #
    #############################################
    def form_matrix(self, sv):
        from numpy import matrix

        n = len(sv)
        listmatrix = []
        for i in range(n):
            row = []
            for j in range(0,i):
                row[j] = listmatrix[j][i]
            for j in range(i,n):                #only compute for upper triangle
                row[j] = self.kernel.k(sv[i].getFeature(), sv[j].getFeature()) 
            listmatrix[i] = row

        nmatrix = matrix(listmatrix)            #should make 2d array into numpy matrix
        return nmatrix

    #############################################
    #                                           #
    #   forms label vector for multiplication   #
    #                                           #
    #############################################
    def form_label_vec(self, sv):
        from numpy import array
        
        ra = []
        for ele in sv:
            if (ele.getLabel().getLabel() == 0):
                ra.append(-1)
            else:
                ra.append(ele.getLabel().getLabel())

        return array(ra)

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

