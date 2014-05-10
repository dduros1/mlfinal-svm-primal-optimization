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
        self.testdict = {}
        self.probabilitydict = {}


    def predict(self, instance):
        signvals = []
        for weight in self.weights:
            signvals.append(self.sign(instance,weight))
        try:
            probabilities = self.probabilitydict[tuple(signvals)]
            most_likely = probabilities.index(max(probabilities))
            return Label(most_likely)
        except Exception:
            pass
        
    def finish(self, instances):
        for inst in instances:
            signvals = []
            for weight in self.weights:
                signvals.append(self.sign(inst,weight))
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
            #print signval, ':', self.testdict[signval]
            #print signval, problist

    def test(self):
        for signval in self.probabilitydict.keys():
            problist = self.probabilitydict[signval]
            print signval, problist


    def train(self, instances):
        separated_instances = self.filter_by_label(instances)

        for insts in separated_instances:
            self.weights.append(self.optimizer.train(insts))
            self.optimizer.clear()
        #Train classifiers for label pairs (1,2), (2,3), (3,4), (4,5)

        self.finish(instances)

    def sign(self, instance, weight):
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
    reader = DataReader('data/train.tsv', punct=1, binary=1)
    reader.readInput()
    data = reader.getData()
    numtest = int(.1*len(data))
    tester = MulticlassSVM(GradientDescent())
    tester.train(data[numtest:])

    for inst in data[:numtest]:
        tester.predict(inst)
    tester.test()
     
    

if __name__ == "__main__":
    main()

    

