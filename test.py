#!/usr/bin/python


from dataparser import *
from data import *
import time
import numpy
import operator
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--multi", "-m", type = int, choices = [0, 1], required = True, help = "0 for binary, 1 for multi")
    parser.add_argument("--test", "-t", default = 0, type = int, choices = [0, 1], help = "0 for labeled data, 1 for not")
    parser.add_argument("--punct", "-p", default = 1, type = int, choices = [0, 1], help = "0 for keep punctuation, 1 for ignore")
    parser.add_argument("--count", "-c", default = 0, type = int, choices = [0, 1], help = "0 for no count, 1 for count")
    parser.add_argument("--lower", "-l", default = 1, type = int, choices = [0, 1], help = "0 for no lower case, 1 for all lower case")
    parser.add_argument("--file", "-f", required = True, help = "datafile")

    args = vars(parser.parse_args())

    reader = new DataReader(args["file"], opt = args["count"], test = args["test"], punct = args["punct"], binary = args["multi"], lower = args["lower"])
    reader.readInput()
    data = reader.getData()


    GradientTest()
    StochasticTest()
    NewtonTest()


def GradientTest():
    pass

def StochasticTest():
    pass

def NewtonTest():
    pass



if __name__ == "__main__":
    main()
