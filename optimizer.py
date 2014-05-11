#!/usr/bin/python

import copy
from data import *
import time
from dataparser import *
import math
from numpy import matrix
from numpy import array
import numpy
import random


class Optimizer:
    weights = Feature()
    

    def __init__(self):
        self.weights = Feature()
        self.basis = 0.0

    def clear(self):
        self.weights = Feature()

    def calc_basis(self, instances):
        count = 0.0
        val = 0.0
        for inst in instances:
            guess = self.weights.dot(inst.getFeature())
            if guess * inst.getLabel().getLabel() < 1:
                count += 1
                val += guess - inst.getLabel().getLabel()
        self.basis = val / count


#############################################
#                                           #
#       Gradient Descent Optimizer          #
#                                           #
#############################################
class GradientDescent(Optimizer):

    #weights = Feature()

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
#       Newton Primal Optimizer             #
#                                           #
#############################################
class NewtonApproximation(Optimizer):

    #weights = Feature()

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
        start = time.time()
        beta = self.primalsvm(instances)
        #weights = sum beta * inst
        for inst in instances:
            for word in inst.getWords():
                try:
                    self.weights.add(word, self.weights.get(word) + beta[inst] * inst.getFeature().get(word))
                except KeyError:
                    pass
        end = time.time()
        print ('it finished!!!', (end-start), 'seconds')
        return self.weights

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
            start = time.time()
            beta = self.primalsvm(small_instances)
            end = time.time()
            print('Seconds for recursive call:', (end-start))
            for key, value in beta.iteritems():
                if not value == 0:
                    sv.append(key)
        else:
            sv = copy.copy(instances)
        #oldsvs = []
        oldoldsv = []                           #check 2 back for oscillation TODO cycles
        oldsv = []
        #while not sv in oldsvs:
        loopcounter = 0
        while not oldsv == sv and not oldoldsv == sv:
            loopstart = time.time()
            #oldsvs.append(sv)
            oldoldsv = copy.copy(oldsv)
            oldsv = copy.copy(sv)

            start = time.time()
            kmatrix = self.form_matrix(sv)
            end = time.time()
            #print ((end-start), 'seconds to form kmatrix')

            tempmatrix = copy.copy(kmatrix)
            for i in range(len(sv)):
                tempmatrix[(i,i)] += self.huberparam
            inverse = tempmatrix.I
            labels = self.form_label_vec(sv)
            beta = inverse.dot(labels)
            start = time.time()
            sv = self.update(instances, beta, kmatrix, oldsv)
            end = time.time()
            #print ((end-start), 'seconds to multiply stuff')
            #print(len(sv))
            loopend=time.time()
            #print((loopend-loopstart), 'seconds for one iteration')
            loopcounter += 1
            if loopcounter % 100 == 0:
                print ('Iteration of loop:', loopcounter)
    
        return self.formdictionary(beta, sv)


    #############################################
    #############################################
    def formdictionary(self, beta, sv):
        mydict = {}
        for i in range(len(beta[0])):
            mydict[sv[i]] = beta[(0,i)]
        return mydict

    #############################################
    #                                           #
    #           update sv step                  #
    #                                           #
    #############################################
    def update(self, instances, beta, kmatrix, oldsv):
        newsv = []
        for i in range(len(instances)):
           val = 0.0
           for j in range(len(oldsv)):
               val += self.kernel.K(instances[i].getFeature(), oldsv[j].getFeature()) * beta[(0,j)]
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
        #from numpy import matrix

        n = len(sv)
        listmatrix = [[]] * n
        for i in range(n):
            row = [0] * n
            for j in range(0,i):
                row[j] = listmatrix[j][i]
            for j in range(i,n):                #only compute for upper triangle
                row[j] = self.kernel.K(sv[i].getFeature(), sv[j].getFeature()) 
            listmatrix[i] = row

        nmatrix = matrix(listmatrix)            #should make 2d array into numpy matrix
        return nmatrix

    #############################################
    #                                           #
    #   forms label vector for multiplication   #
    #                                           #
    #############################################
    def form_label_vec(self, sv):
        #from numpy import array
        
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
    
    def __init__(self, sigma=2, caching = 0):
        self.sigma = sigma
        self.caching = caching #1 if caching, 0 if not
        self.cache = {}

    #x1, x2 are Features
    def K(self, x1, x2):
        #import math
        if self.caching:
            id1 = x1.getID()
            id2 = x2.getID()
            if id1 not in self.cache:
                self.cache[id1] = {}
            if id2 not in self.cache[id1]:
                self.cache[id1][id2] =  math.exp(-1*x1.squared_norm(x2)/(2*self.sigma*self.sigma))
            return self.cache[id1][id2]
        else:
            return math.exp(-1*x1.squared_norm(x2)/(2*self.sigma*self.sigma))

#############################################
#                                           #
#       Gradient Descent Optimizer          #
#                                           #
#############################################
class StochasticSubgradient(Optimizer):
        
    #weights = Feature()

    def __init__(self, param = 1.0, iterations = 25, sample_portion = 10):
        self.param = param
        self.iterations = iterations
        self.sample_portion = sample_portion

    def train(self, instances):
        w = Feature()
        for t in range(1, self.iterations+1):
            #if t % 10 == 0:
            #    print('iteration', t)
            A = random.sample(instances, self.sample_portion)            
            newA = []            
            for inst in A:
                if inst.getLabel().getLabel() == 0:
                    w.scalar_multiply(-1)
                if w.dot(inst.getFeature()) < 1:
                    newA.append(inst)
                #undo the negativity, if needed.
                if inst.getLabel().getLabel() == 0:
                    w.scalar_multiply(-1)                
            eta = 1.0 / (self.param * t)
            w.scalar_multiply(1 - eta * self.param)
            for inst in newA:
                x = copy.copy(inst.getFeature())
                x.scalar_multiply(eta / self.sample_portion)
                if inst.getLabel().getLabel() == 0:
                    x.scalar_multiply(-1)
                w.plusall(x)
            norm = 1.0 / (w.self_norm() * math.sqrt(self.param))
            if norm < 1:
                w.scalar_multiply(norm)

        self.weights = Feature()
        return w

#############################################
def main():
    reader = DataReader('data/smalltrain.tsv', punct=1)
    reader.readInput()
    data = reader.getData()
    print('Data Read :)')
    tester = NewtonApproximation(RBF(caching=1))
    tester.train(data)

if __name__ == "__main__":
    main()

