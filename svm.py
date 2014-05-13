#!/usr/bin/python


from optimizer import *
from data import *
import time

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
        self.basis = self.optimizer.calc_basis(instances)

    #############################################
    #                                           #
    #   gives which side of the hyper plane     #
    #     an instance should be on              #
    #                                           #
    #############################################
    def sign(self, instance):
        if instance.getFeature().dot(self.weights) + self.basis >= 0:
            return 1
        return -1


#############################################
#                                           #
#   This class implements a linear SVM      #
#       for multiclass data                 #
#                                           #
#############################################
class MulticlassSVM(SVM):

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.weights = []       ##List of features
        self.basises = []
        self.testdict = {}
        self.probabilitydict = {}
        self.totalprobabilities = [0,0,0,0,0]

    #############################################
    #                                           #
    #   given a list of training instances      #
    #                                           #
    #############################################
    def predict(self, instance):
        signvals = []
        for index in range(len(self.weights)):
            signvals.append(self.sign(instance,self.weights[index], self.basises[index]))
        try: #most likely given signval combo
            probabilities = self.probabilitydict[tuple(signvals)]
            most_likely = probabilities.index(max(probabilities))
        except Exception:  #if we haven't seen that combination of signvals, overall most likely
            most_likely = self.totalprobabilities.index(max(self.totalprobabilities))            
        return Label(most_likely)
        
    #############################################
    #                                           #
    #   Calculate the probability of a label    #
    #     given a sign val and overall label    #
    #     probabilities                         #
    #                                           #
    #############################################
    def finish(self, instances):
        for inst in instances:
            self.totalprobabilities[inst.getLabel().getLabel()] += 1
            signvals = []
            for index in range(len(self.weights)):
                signvals.append(self.sign(inst,self.weights[index], self.basises[index]))
            if not tuple(signvals) in self.testdict.keys():
                tempdict = {}
                tempdict[inst.getLabel().getLabel()] = 1
                self.testdict[tuple(signvals)] = tempdict
            else:
                if not inst.getLabel().getLabel() in self.testdict[tuple(signvals)]:
                    self.testdict[tuple(signvals)][inst.getLabel().getLabel()] = 1
                else:
                    self.testdict[tuple(signvals)][inst.getLabel().getLabel()]+=1

        #Compute probability of label given signval, pick highest prob as label
        for signval, labeldict in self.testdict.iteritems():
            problist = [0,0,0,0,0]
            total = sum(labeldict.values())
            for label, count in labeldict.iteritems():
                problist[label] = float(count)/total
                
            self.probabilitydict[signval] = problist

        #Compute overall probability of label (via counts)
        for ele in self.totalprobabilities:
            ele = ele/sum(self.totalprobabilities)



    #############################################
    #                                           #
    #   given a list of training instances,     #
    #     train pairwise svms with basis fn     #
    #                                           #
    #############################################
    def train(self, instances):
        separated_instances = self.filter_by_label(instances)

        #Train classifiers for label pairs (1,2), (2,3), (3,4), (4,5)
        for insts in separated_instances:
            start = time.time()
            self.weights.append(self.optimizer.train(insts))
            next = time.time()
            #print('weights: %f seconds' % (next -start))
            self.basises.append(self.optimizer.calc_basis(insts))
            next2 = time.time()
            #print('basis: %f seconds' % (next2 - next))
            #self.basises.append(0)
            self.optimizer.clear()
            end = time.time()
            #print ('clear: %f seconds' % (end - next2))
        
        self.finish(instances)

    #############################################
    #                                           #
    #   primal svm constraint (hyperplane)      #
    #                                           #
    #############################################
    def sign(self, instance, weight, basis):
        if instance.getFeature().dot(weight) + basis >= 0:
            return 1
        return -1

    #############################################
    #                                           #
    #  create training sets for pairwise svms   #
    #                                           #
    #############################################
    def filter_by_label(self, instances):
        list1=[]
        list2=[]
        list3=[]
        list4=[]
        for inst in instances:
            val = inst.getLabel().getLabel()
            if val == 1:
                list1.append(Instance(Label(0), inst.getFeature()))
            elif val == 2:
                list1.append(Instance(Label(1), inst.getFeature()))
                list2.append(Instance(Label(0), inst.getFeature()))
            elif val == 3:
                list2.append(Instance(Label(1), inst.getFeature()))
                list3.append(Instance(Label(0), inst.getFeature()))
            elif val == 4:
                list3.append(Instance(Label(1), inst.getFeature()))
                list4.append(Instance(Label(0), inst.getFeature()))
            elif val == 5:
                list4.append(Instance(Label(1), inst.getFeature()))
        filtered_list = [list1, list2, list3, list4]
        return filtered_list
##################################### TESTING #########################################

def main():
    reader = DataReader('data/largetrain.tsv', punct=1, binary=1)
    reader.readInput()
    data = reader.getData()
    numtest = int(.1*len(data))
    #tester = SVM(GradientDescent())
    tester = MulticlassSVM(GradientDescent())
    tester.train(data[numtest:])

    correct = 0
    for inst in data[:numtest]:
        newlabel = tester.predict(inst)
        if newlabel.equals(inst.getLabel()):
            correct += 1
    #tester.test()
    print 'accuracy', (float(correct)/len(data[:numtest]))
     
    

if __name__ == "__main__":
    main()

    

