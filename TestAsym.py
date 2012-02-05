#!/usr/bin/env python
# encoding: utf-8
"""
TestAsym.py

Created by Derek Koh on 2011-07-19.

"""
import AsymForest, csv, math, operator
# 
# data, features = AsymForest.read_data("StephenMarsland/iris.csv")

# AsymForest.randomForest(data,10)


outputwriter=csv.writer(open('AsymmetricResults.csv', 'wb'))
data, features = AsymForest.read_data("Asymmetricdata.csv",0,0)
simulationparam=[10,100,1000] # number of trees to use
for nooftrees in simulationparam:
	results=AsymForest.randomForest(data,nooftrees)
	outputrow=[nooftrees]
	sorted_results = sorted(results.iteritems(), key=operator.itemgetter(1),reverse=True)
	for value in sorted_results:
		outputrow.append(value[0])
	outputwriter.writerow(outputrow)
# 
# rows=data[1:]
# for column in xrange(10):
# 	current_score=AsymForest.giniimpurity(rows,0)
# 	(set1,set2)=AsymForest.divideset(rows,column,1)
# 	# Information gain
# 	p=float(math.log(len(set1)))/float(math.log(len(set1))+math.log(len(set2)))
# 	leftnodepurity=AsymForest.giniimpurity(set1,1)
# 	if leftnodepurity == -99:
# 		leftnodepurity=current_score
# 	rightnodepurity=AsymForest.giniimpurity(set2,1)
# 	if rightnodepurity==-99:
# 		rightnodepurity=current_score
# 	gain=current_score-p*leftnodepurity-(1-p)*rightnodepurity
# 	print column, current_score, p, leftnodepurity, rightnodepurity, gain
