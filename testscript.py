#!/usr/bin/env python
# encoding: utf-8
"""
testscript.py

Created by Derek Koh on 2011-07-07.
Copyright (c) 2011 __MyCompanyName__. All rights reserved.
"""

import random, Forest, math, treepredictstratum




# Code to test Forest.py
#dummydata, featureNames=Forest.read_data('StephenMarsland/breast.csv',1)
#Forest.randomForest(dummydata,50)

#This code transform stratum data to its independent variables
def transformstratum(data):
	returndata = data[:] #create a copy of input data
	column_count=len(data[0])
	column_count_half=column_count/2
	for row in returndata:
		for i in range(column_count_half):
			if row[i] < row[i+column_count_half]: 
				row[i+column_count_half]=1
			else: row[i+column_count_half]=0
	return returndata


# def getanswer(data):
# 	returndata = data[:] #create a copy of input data
# 	column_count=len(data[0])
# 	column_count_half=column_count/2
# 	for row in returndata:
# 		for i in range(column_count_half):
# 			if row[i] < row[i+column_count_half]: 
# 				row[i+column_count_half]=1
# 			else: row[i+column_count_half]=0
# 	return returndata


dummydata, featureNames=Forest.read_data('dummydata_10.csv')
dummydata1=[[int(random.random()*100) for i in xrange(20)] for j in xrange(len(dummydata))]
for i in xrange(len(dummydata)):
	dummydata1[i][0]=dummydata[i][0]
	dummydata1[i][1]=dummydata[i][1]
	dummydata1[i][len(dummydata1[0])/2]=dummydata[i][2]
	dummydata1[i][len(dummydata1[0])/2+1]=dummydata[i][3]



dummydata=transformstratum(dummydata)
dummydata1=transformstratum(dummydata1)

#for row in dummydata: print row[2:4]
sim_mat= treepredictstratum.randomForest(dummydata,50)
for row in sim_mat: print row
newcluster=treepredictstratum.hcluster(sim_mat)
#treepredictstratum.printclust(newcluster)
#treepredictstratum.randomForest(dummydata,500)






# primes = [x for x in range(2, 50) if x not in noprimes]


