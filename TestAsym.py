#!/usr/bin/env python
# encoding: utf-8
"""
TestAsym.py

Created by Derek Koh on 2011-07-19.

"""
import AsymForest
# 
data, features = AsymForest.read_data("StephenMarsland/iris.csv")


# blah=[[0]]*4 + [[1]]*6
# print AsymForest.giniimpurity(blah)


AsymForest.randomForest(data,10)

# 
# 
# from scipy.io import arff
# data, meta = arff.loadarff('arffFiles/wine.arff')
# print meta
# print len(data)
