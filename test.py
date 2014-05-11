#!/usr/bin/python


from dataparser import DataReader
import time
import operator
import argparse
from optimizer import *
from crossvalidation import CrossValidationTester


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--multi", "-m", type = int, choices = [0, 1], required = True, help = "0 for binary, 1 for multi")
    parser.add_argument("--test", "-t", default = 0, type = int, choices = [0, 1], help = "0 for labeled data, 1 for not")
    parser.add_argument("--punct", "-p", default = 1, type = int, choices = [0, 1], help = "0 for keep punctuation, 1 for ignore")
    parser.add_argument("--count", "-c", default = 0, type = int, choices = [0, 1], help = "0 for no count, 1 for count")
    parser.add_argument("--lower", "-l", default = 1, type = int, choices = [0, 1], help = "0 for no lower case, 1 for all lower case")
    parser.add_argument("--file", "-f", required = True, help = "datafile")
    parser.add_argument("--optimizer", "-o", required = True, choices = ["gradient", "stochastic", "newton"], help = "choice of optimizer")

    args = vars(parser.parse_args())

    reader = DataReader(args["file"], opt = args["count"], test = args["test"], punct = args["punct"], binary = args["multi"], lower = args["lower"])
    reader.readInput()
    data = reader.getData()

    if args["test"] == 0:        
        if args["optimizer"] == "gradient":
            GradientTest(data, args["multi"])
        elif args["optimizer"] == "stochastic":        
            StochasticTest(data, args["multi"])
        elif args["optimizer"] == "newton":    
            NewtonTest(data, args["multi"])

def GradientTest(instances, multi):

    best_result = 0.0
    best_learn_rate = 0.0
    for i in [3, 2, 1]:
        for j in [1, 2, 3, 4, 5]:
            start = time.time()
            learn_rate = j * (10 ** (-1 * i))
            optimizer = GradientDescent(learning_rate = learn_rate, iterations = 10)    
            tester = CrossValidationTester(instances, optimizer, multi)
            result = tester.runtest()
            end = time.time()            
            print ('Learn rate: %f; accuracy: %f' % (learn_rate, result))        
            print ('Time taken: %d seconds' % (end - start))
            if result > best_result:
                best_result = result
                best_learn_rate = learn_rate

    print ('Best learn rate: %f; accuracy: %f' % (best_learn_rate, best_result))

    best_result = 0.0
    best_iterations = 0
    for iterations in range(10, 51, 5):
        start = time.time()
        optimizer = GradientDescent(learning_rate = best_learn_rate, iterations = iterations)
        tester = CrossValidationTester(instances, optimizer, multi)
        result = tester.runtest()
        end = time.time()
        print ('Iterations: %d; accuracy: %f' % (iterations, result))
        print ('Time taken: %d seconds' % (end - start))
        if result > best_result:
            best_result = result
            best_iterations = iterations

    print ('Best number of iterations: %d; accuracy: %f' % (best_iterations, best_result))


def StochasticTest(instances, multi):
    best_result = 0.0
    best_param = 0
    for i in [-3, -2, -1, 0, 1]:
        start = time.time()        
        diam_param = 10 ** i
        optimizer = StochasticSubgradient(param = diam_param)
        tester = CrossValidationTester(instances, optimizer, multi)
        result = tester.runtest()
        end = time.time()
        print ('Diameter parameter: %f; accuracy: %f' % (diam_param, result))
        print ('Time taken: %d seconds' % (end - start))
        if result > best_result:
            best_result = result
            best_param = diam_param

    print ('Best diameter parameter: %f; accuracy: %f' % (best_param, best_result))
    
    best_sample_size = 0
    best_result = 0.0
    for size in range(10, 201, 10):
        if size > len(instances) * 0.9:
            break
        start = time.time()
        optimizer = StochasticSubgradient(param = best_param, sample_portion = size)
        tester = CrossValidationTester(instances, optimizer, multi)
        result = tester.runtest()
        end = time.time()
        print ('Sample size: %d; accuracy: %f' % (size, result))
        print ('Time taken: %d seconds' % (end - start))
        if result > best_result:
            best_sample_size = size
            best_result = result

    print ('Best sample size: %d; accuracy: %f' % (best_sample_size, best_result))

    best_iters = 0
    best_result = 0.0
    for iterations in range(10, 201, 10):
        start = time.time()
        optimizer = StochasticSubgradient(param = best_param, sample_portion = best_sample_size, iterations = iterations)
        tester = CrossValidationTester(instances, optimizer, multi)
        result = tester.runtest()
        end = time.time()
        print ('Iterations: %d; accuracy: %f' % (iterations, result))
        print ('Time taken: %d seconds' % (end - start))
        if result > best_result:
            best_iters = iterations
            best_result = result

    print ('Best iteration number: %d; accuracy: %f' % (best_iters, best_result))

def NewtonTest(instances, multi):

    best_lambda = 0.0
    best_result = 0
    for i in [-6, -5, -4, -3, -2, -1, 0]:
        param = 10 ** i
        start = time.time()
        optimizer = NewtonApproximation(RBF(), param = param, iterations = 10)
        tester = CrossValidationTester(instances, optimizer, multi)
        result = tester.runtest()
        end = time.time()
        print ('Lambda: %f; accuracy: %f' % (param, result))
        print ('Time taken: %d seconds' % (end - start))
        if result > best_result:
            best_result = result
            best_lambda = param
        
    print ('Bester lambda: %f; accuracy: %f' % (best_lambda, best_result))


if __name__ == "__main__":
    main()
