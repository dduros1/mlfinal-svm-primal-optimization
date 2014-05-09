#!/usr/bin/python

from optimizer import *
import pickle
import time
from dataparser import *
from optimizer import *

start = time.time()
reader = DataReader('data/train.tsv', punct=1)
reader.readInput()
data = reader.getData()
tester = NewtonApproximation(RBF())
kernel = {}
for i in data:
    id1 = i.getID()
    kernel[id1] = {}
    for j in data:
        id2 = j.getID()
        kernel[id1][id2] = tester.kernel.K(i.getFeature(), j.getFeature()) 
picklefile = open('precomputedkernel.pkl', 'w')
pickle.dump(kernel, picklefile)
end = time.time()
print('Kernel computed in ', (end-start)/60, ' seconds')
