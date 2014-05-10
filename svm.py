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
        #print instance.getLabel(), signvals

    def train(self, instances):
        separated_instances = self.filter_by_label(instances)
        for insts in separated_instances:
            self.weights.append(self.optimizer.train(insts))
        #Train classifiers for label pairs (1,2), (2,3), (3,4), (4,5)
        for w in self.weights:
            print w.sum()

    def sign(self, instance, weight):
        if instance.getFeature().dot(weight) >= 0:
            return 1
        return -1

    #TODO why are they all the same
    def filter_by_label(self, instances):
        filtered_list = [[]]*4
        count = 0
        count2 = 0
        count3 = 0
        count4 = 0
        for inst in instances:
            print 'one inst'
            val = inst.getLabel().getLabel()
            if val == 1:
                print 'added to 0'
                filtered_list[0].append(inst)
                count+=1
            elif val == 2:
                print 'added to 0, 1'
                filtered_list[0].append(inst)
                filtered_list[1].append(inst)
                count+=1
                count2+=1
            elif val == 3:
                print 'added to 1,2'
                filtered_list[1].append(inst)
                filtered_list[2].append(inst)
                count2+=1
                count3+=1
            elif val == 4:
                print 'added to 2, 3'
                filtered_list[2].append(inst)
                filtered_list[3].append(inst)
                count3+=1
                count4+=1
            elif val == 5:
                print 'added to 3'
                filtered_list[3].append(inst)
                count4+=1
        print len(filtered_list[0]), len(filtered_list[1]), len(filtered_list[2]), len(filtered_list[3])
        print len(instances)
        print filtered_list[0] == filtered_list[1]
        print count, count2, count3, count4
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

    

