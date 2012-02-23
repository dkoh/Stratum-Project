#!/usr/bin/env python
# encoding: utf-8
"""
createdata.py

Created by Derek Koh on 2011-07-10.
Copyright (c) 2011 __MyCompanyName__. All rights reserved.
"""

import random, csv
random.seed(123456780)

def createstratumdata(obs):
	newdata=[['controlX','controlY','caseX','caseY']]
	for i in xrange(obs):
		location=[random.random()*10,random.random()*10,0,0]
		if random.random() < .9:
			if 0 < location[0] <= 5 and 0 < location[1] <=6:
				location[2]=location[0]+random.random()*10
				location[3]=location[1]+random.random()*10	
			elif 0 < location[0] <= 7 and 6 < location[1] <=10:
				location[2]=location[0]+random.random()*-10
				location[3]=location[1]+random.random()*10
			elif 7 < location[0] <= 10 and 3 < location[1] <=10:
				location[2]=location[0]+random.random()*10
				location[3]=location[1]+random.random()*-10
			else:
				location[2]=location[0]+random.random()*-10
				location[3]=location[1]+random.random()*-10
		else:
			location[2]=location[0]+(random.random()-.5)*20
			location[3]=location[1]+(random.random()-.5)*20
			
		newdata.append(location)		
	outputwriter=csv.writer(open('dummydata.csv', 'wb'))
	for row in newdata:
		outputwriter.writerow(row)

# createstratumdata(500)

def createVanilladata(obs):
	newdata=[['controlX','controlY','caseX','caseY']]
	for i in xrange(obs):
		location=[random.random()*10,random.random()*10,0,0]
		if random.random() < .9:
			location[2]=location[0]+random.random()*10
			location[3]=location[1]+random.random()*10	
		else:
			location[2]=location[0]+(random.random()-.5)*20
			location[3]=location[1]+(random.random()-.5)*20	
		newdata.append(location)		
	outputwriter=csv.writer(open('vanillalogisticdata.csv', 'wb'))
	for row in newdata:
		outputwriter.writerow(row)

# createVanilladata(500)

def createAsymmetricdata(obs):	
	Key=[[.9,.9],[.97,.15],[.15,.97],[.9,.5],[.5,.9]] + [[.5,.5] for i in xrange(10)]

	Initial_Conditions=[0]*int(obs/2) +[1]*int(obs/2)

	Dataset = [
	    [i] + [
	        i if random.random() < k[i] else 1 - i
	        for k in Key
	    ]
	    for i in Initial_Conditions
	]

	Key= [str(x[0])+"/"+str(x[1]) for x in Key]
	Dataset=[["Y"]+ Key]+ Dataset	
	outputwriter=csv.writer(open('Asymmetricdata1.csv', 'wb'))
	for row in Dataset:
		outputwriter.writerow(row)

	#printing probabilites of simulated dataset.
	finalprob =dict([(i,(0,0)) for i in xrange(len(Dataset[0]))])
	for i in finalprob.keys():
		if i ==0: continue
		ColumnsProb=[0.0,0.0,0.0,0.0]
		for row in Dataset:
			if row[i]==0 and row[0]==0: ColumnsProb[0]=ColumnsProb[0]+1
			if row[i]==1 and row[0]==0: ColumnsProb[1]=ColumnsProb[1]+1
			if row[i]==0 and row[0]==1: ColumnsProb[2]=ColumnsProb[2]+1
			if row[i]==1 and row[0]==1: ColumnsProb[3]=ColumnsProb[3]+1
		finalprob[i]=(ColumnsProb[3]/(ColumnsProb[3]+ColumnsProb[1]),ColumnsProb[0]/(ColumnsProb[0]+ColumnsProb[2]))
	for i in finalprob:
		print i, finalprob[i]

#Used for figuring out the characteristics of data 
#confusionMatrix(map(operator.itemgetter(0), data) ,map(operator.itemgetter(len(features)-1), data))
def confusionMatrix(predictor, response, label=1):
	returnmatrix=[0.0]*4
	for i in xrange(len(predictor)):
		if i== 0 and label==1: continue
		if predictor[i]==1 and response[i]==1:
			returnmatrix[0]=returnmatrix[0]+1
		if predictor[i]==1 and response[i]==0:
			returnmatrix[1]=returnmatrix[1]+1
		if predictor[i]==0 and response[i]==1:
			returnmatrix[2]=returnmatrix[2]+1
		if predictor[i]==0 and response[i]==0:
			returnmatrix[3]=returnmatrix[3]+1
	print returnmatrix
	summarymatrix= [returnmatrix[0]/(returnmatrix[0]+returnmatrix[1]),returnmatrix[1]/(returnmatrix[0]+returnmatrix[1]),returnmatrix[2]/(returnmatrix[2]+returnmatrix[3]),returnmatrix[3]/(returnmatrix[2]+returnmatrix[3])]		
	summarymatrix=[round(i,2) for i in summarymatrix]
	print summarymatrix

#creates data with column 1 and 2 interaction creating a highly asymmetric power variable
def asymmetricDatawithInteraction(obs):
	Initial_Conditions=[0]*obs
	testdata=[[0]+ [round(random.random()) for j in xrange(10)] for i in Initial_Conditions]
	for row in testdata: #build interacting variables
		if row[1]==1 and row[2]==1 and random.random() < .95: row[0]=1
		else: row[0] = round(random.random())
	for row in testdata:
		if row[0]==1 : 
			if random.random() <.9: 
				row[3]=1
			else:
				row[3]=0
		else:
			if random.random() <.6: 
				row[3]=0
			else:
				row[3]=1
		if row[0]==1 : 
			if random.random() <.5: 
				row[4]=1
			else:
				row[4]=0
		else:
			if random.random() <.8: 
				row[4]=0
			else:
				row[4]=1
	outputwriter=csv.writer(open('Asymmetricdata2.csv', 'wb'))
	for row in testdata:
		outputwriter.writerow(row)
	return testdata

#random code
testdata=asymmetricDatawithInteraction(1000)
#createAsymmetricdata(1000)

#code to verify asymmetric data with interactions
# import operator
# confusionMatrix(map(operator.itemgetter(1), testdata) ,map(operator.itemgetter(0), testdata),0)
# confusionMatrix(map(operator.itemgetter(2), testdata) ,map(operator.itemgetter(0), testdata),0)
# confusionMatrix(map(operator.itemgetter(3), testdata) ,map(operator.itemgetter(0), testdata),0)
# confusionMatrix(map(operator.itemgetter(4), testdata) ,map(operator.itemgetter(0), testdata),0)
# for row in testdata:
# 	if row[1]==1 and row[2]==1:
# 		row[3]=1
# 	else:
# 		row[3]=0
# confusionMatrix(map(operator.itemgetter(3), testdata) ,map(operator.itemgetter(0), testdata),0)
	



