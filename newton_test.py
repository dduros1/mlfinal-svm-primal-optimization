from dataparser import *
from optimizer import *
from svm import *
import random
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi", "-m", type = int, choices = [0, 1], required = True, help = "0 for binary, 1 for multi")
    parser.add_argument("--lambda", "-l", type = float, default = 0.001, help = "lambda parameter")
    parser.add_argument("--file", "-f", required = True, help = "datafile")

    args = vars(parser.parse_args())

    reader = DataReader(args["file"], binary = args["multi"])
    reader.readInput()
    data = reader.getData()

    optimizer = NewtonApproximation(RBF(caching = 1), param = args["lambda"])
    random.shuffle(data)
    test = data[:int(len(data) * 0.1) + 1]
    train = data[int(len(data) * 0.1) + 1:]
    if args["multi"] == 0:
        mysvm = SVM(optimizer)
    else:
        mysvm = MulticlassSVM(optimizer)
    start = time.time()
    mysvm.train(train)
    end = time.time()
    correct = 0.0
    for instance in test:
        newlabel = mysvm.predict(instance)
        if newlabel.equals(instance.getLabel()):
            correct += 1
    
    print('Accuracy: %f' % (correct / len(test)))
    print('Time taken: %d' % (end - start))

if __name__ == "__main__":
    main()
