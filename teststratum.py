#!/usr/bin/env python
# encoding: utf-8
"""
teststratum.py
Created by Derek Koh on 2011-07-07.
Copyright (c) 2011 __MyCompanyName__. All rights reserved.
"""

import random, Forest1, math, clusters
from copy import deepcopy 
import rpy2.robjects as robjects
random.seed(2222222)


inputdict={'readDataset':("vanillapaireddata.csv",0,-99),'number_of_trees':10}
blah=Forest1.ConditionalRandomForest(**inputdict)
#
#def logisticregressionR(data):
#	data1=zip(*data)
#	features=['col{0}'.format(i) for i in xrange(len(data[0]))]
#	columns=[robjects.FloatVector(col) for col in data1]
#	Rdata = robjects.r['data.frame'](**dict(zip(features,columns)))
#	Rformula =  robjects.r['as.formula']('{0} ~ {1} -1'.format(features[-1],reduce(lambda x,y: x + '+' +  y, features[:-1] )))
#	rpart_params = {'formula' : Rformula, 'data' : Rdata, 'family' : "binomial"}
#	model=robjects.r.glm(**rpart_params)
#	return model.rx('aic')[0][0],model.rx('deviance')[0][0]
#
###This function transform stratum data to its independent variables
#def transformstratum(data, clrformat=0):  
#	returndata = deepcopy(data) #create a copy of input data
#	column_count=len(data[0])
#	column_count_half=column_count/2
#	if clrformat ==0:
#		for row in returndata:
#			for i in range(column_count_half):
#				if row[i] < row[i+column_count_half]: 
#					row[i+column_count_half]=1
#				else: row[i+column_count_half]=0
#	else:
#		for row in returndata:
#			for i in range(column_count_half):
#				row[i+column_count_half]=row[i+column_count_half]-float(row[i])				
#		returndata=[row[column_count_half:] + [1] for row in returndata]
#	return returndata
#
#logisticdata= Forest1.read_data("vanillapaireddata.csv",0,-99)[0]
#logisticdata=transformstratum(logisticdata,1)
#import csv 
#outputwriter=csv.writer(open('Rlogisticdata.csv', 'wb'))
#for row in logisticdata:
#	outputwriter.writerow(row)
#print logisticregressionR(logisticdata)
#print "done"
##This code does plain CLR on a dataset and gets back the aic and deviance
#def logisticregressionR(data):
#	data1=zip(*data)
#	features=['col{0}'.format(i) for i in xrange(len(data[0]))]
#	columns=[robjects.FloatVector(col) for col in data1]
#	Rdata = robjects.r['data.frame'](**dict(zip(features,columns)))
#	Rformula =  robjects.r['as.formula']('{0} ~ {1} -1'.format(features[-1],reduce(lambda x,y: x + '+' +  y, features[:-1] )))
#	rpart_params = {'formula' : Rformula, 'data' : Rdata, 'family' : "binomial"}
#	model=robjects.r.glm(**rpart_params)
#	return model.rx('aic')[0][0],model.rx('deviance')[0][0]
#		
######################
##START OF MAIN SCRIPT
######################
#numberofnodes=10
#
##The input dataset as 4 columns. the xy of the case and xy for the control
#rawdata, featureNames=Forest.read_data('vanillalogisticdata.csv')
#logisticdata=treepredictstratum.transformstratum(rawdata,1)
#clusters= treepredictstratum.stratumForest(treepredictstratum.transformstratum(rawdata),numberofnodes)
#
## Perform logistic regression in each of the 10 clusters and then suming up the stats
#finallist=[]
#for key in clusters:
#	if len(clusters[key])>3: 
#		finallist.append(logisticregressionR([logisticdata[i] for i in clusters[key]]))
#	else: print 'Less than 3 obs in cluster'
#
##print out the final results for the number of nodes. 
#print "Results for {0} nodes".format(numberofnodes)
#print reduce(lambda x, y: (x[0]+y[0],x[1]+y[1]),finallist)
#
##Calculating the vanilla logistic regression 
#print treepredictstratum.logisticregressionR(logisticdata)	
	





# Unused code
# dummydata1=[[int(random.random()*100) for i in xrange(20)] for j in xrange(len(dummydata))]
# for i in xrange(len(dummydata)):
# 	dummydata1[i][0]=dummydata[i][0]
# 	dummydata1[i][1]=dummydata[i][1]
# 	dummydata1[i][len(dummydata1[0])/2]=dummydata[i][2]
# 	dummydata1[i][len(dummydata1[0])/2+1]=dummydata[i][3]

