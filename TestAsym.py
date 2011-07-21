#!/usr/bin/env python
# encoding: utf-8
"""
TestAsym.py

Created by Derek Koh on 2011-07-19.
Copyright (c) 2011 __MyCompanyName__. All rights reserved.
"""
import AsymForest

#data, features = AsymForest.read_data("StephenMarsland/iris.csv")
#print data



from scipy.io import arff
data, meta = arff.loadarff('arffFiles/wine.arff')
print meta
print len(data)
#AsymForest.randomForest(data,1)