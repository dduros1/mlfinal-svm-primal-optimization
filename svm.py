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
        self.optimizer = optimizer
        self.weights = []       ##List of features


    def predict(self, instance):
        signvals = []
        for weight in self.weights:
            signvals.append(self.sign(instance,weight))
        print instance.getLabel(), signvals
        #TODO all are getting ones....

    def train(self, instances):
        separated_instances = self.filter_by_label(instances)

        for insts in separated_instances:
            self.weights.append(self.optimizer.train(insts))
            self.optimizer.clear()
        #Train classifiers for label pairs (1,2), (2,3), (3,4), (4,5)

    def sign(self, instance, weight):
        print instance.getFeature().dot(weight)

        if instance.getFeature().dot(weight) >= 0:
            return 1
        return -1

    def filter_by_label(self, instances):
        list1=[]
        list2=[]
        list3=[]
        list4=[]
        for inst in instances:
            val = inst.getLabel().getLabel()
            if val == 1:
                list1.append(inst)
            elif val == 2:
                list1.append(inst)
                list2.append(inst)
            elif val == 3:
                list2.append(inst)
                list3.append(inst)
            elif val == 4:
                list3.append(inst)
                list4.append(inst)
            elif val == 5:
                list4.append(inst)
        filtered_list = [list1, list2, list3, list4]
        return filtered_list
##################################### TESTING #########################################

def main():
    reader = DataReader('data/smalltrain.tsv', punct=1, binary=1)
    reader.readInput()
    data = reader.getData()
    numtest = int(.1*len(data))
    tester = MulticlassSVM(GradientDescent())
    tester.train(data[numtest:])

    for inst in data[:numtest]:
        tester.predict(inst)
     
    

if __name__ == "__main__":
    main()

    

