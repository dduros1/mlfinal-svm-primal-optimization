import optimizer
import pickle
import time


start = time.time()
reader = DataReader('data/train.tsv', punct=1)
reader.readInput()
data = reader.getData()
kernel = optimizer.computekernel(data)
picklefile = open('precomputedkernel.pkl', 'rw')
pickle.dump(pickelfile, kernel)
end = time.time()
print('Kernel computed in ', (end-start)/60, ' seconds')