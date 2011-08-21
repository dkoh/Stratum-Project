#!/usr/bin/env python
# encoding: utf-8
"""
TestAsym.py

Created by Derek Koh on 2011-07-19.
Copyright (c) 2011 __MyCompanyName__. All rights reserved.
"""
import AsymForest
# 
data, features = AsymForest.read_data("StephenMarsland/iris.csv")
AsymForest.randomForest(data,10)

# 
# 
# from scipy.io import arff
# data, meta = arff.loadarff('arffFiles/wine.arff')
# print meta
# print len(data)
