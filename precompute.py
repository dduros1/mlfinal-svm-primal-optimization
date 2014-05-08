import optimizer
import pickle
import time
from dataparser import *
from optimizer import *

start = time.time()
reader = DataReader('data/train.tsv', punct=1)
reader.readInput()
data = reader.getData()
tester = NewtonApproximation(RBF())
kernel = tester.compute_kernel(data)
picklefile = open('precomputedkernel.pkl', 'rw')
pickle.dump(picklefile, kernel)
end = time.time()
print('Kernel computed in ', (end-start)/60, ' seconds')
