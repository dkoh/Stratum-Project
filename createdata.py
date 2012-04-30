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

def createstratumdata2(obs):
	vectorspace=10
	control_data=[[random.random()*10 for i in xrange(vectorspace)] for n in xrange(obs)]
	case_data=[[0 for i in xrange(vectorspace)] for n in xrange(obs)]
	#Gamma sets the precision of the observation. (precision, plus or negative 1)
	gamma=[(.9,1)]+[(.5,1)]*9
	if len(gamma) !=  vectorspace: print "WARNING: CHECK GAMMA LENGTH"
	finaldata=[]
	for i in xrange(obs):
		for j in xrange(vectorspace):
			if random.random() < gamma[j][0]: case_data[i][j]=control_data[i][j]+ .1*gamma[j][1]
			else: case_data[i][j]=control_data[i][j] - .1*gamma[j][1]
		finaldata.append(case_data[i]+control_data[i])
	outputwriter=csv.writer(open('vanillapaireddata.csv', 'wb'))
	labels=["a1","a2","a3","a4","a5","a6","a7","a8","a9","a10","b1","b2","b3","b4","b5","b6","b7","b8","b9","b10"]
	outputwriter.writerow(labels)
	for row in finaldata:
		outputwriter.writerow(row)
		
			
		 
	subspaces_count=10
	relevant_vector=5
	blah=[[(0,random.random()) for i in xrange(relevant_vector)] for i in xrange(subspaces_count)]
	blah[1][1]=2
	blah[3][4]=7


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
	Key=[[.9,.9],[1,.15],[.9,.5],[.8,.5],[.8,.6],[.7,.5],[.6,.5],[.15,1],[.5,.9],[.5,.8],[.6,.8],[.5,.7],[.5,.6]] + [[.5,.5] for i in xrange(17)]

	Initial_Conditions=[0]*int(obs/2) +[1]*int(obs/2)

	Dataset = [
	    [i] + [
	        i if random.random() < k[i] else 1 - i
	        for k in Key
	    ]
	    for i in Initial_Conditions
	]
	

	#printing Precision of simulated dataset.
	finalprob =dict([(i,(0,0)) for i in xrange(len(Dataset[0]))])
	for i in finalprob.keys():
		if i ==0: continue
		ColumnsProb=[0.0,0.0,0.0,0.0]
		for row in Dataset:
			if row[i]==0 and row[0]==0: ColumnsProb[0]=ColumnsProb[0]+1
			if row[i]==1 and row[0]==0: ColumnsProb[1]=ColumnsProb[1]+1
			if row[i]==0 and row[0]==1: ColumnsProb[2]=ColumnsProb[2]+1
			if row[i]==1 and row[0]==1: ColumnsProb[3]=ColumnsProb[3]+1
		finalprob[i]=(round(ColumnsProb[3]/(ColumnsProb[3]+ColumnsProb[1]),2),round(ColumnsProb[0]/(ColumnsProb[0]+ColumnsProb[2]),2))
	for i in finalprob:
		print i, finalprob[i]
		
	#Getting probability
	Key= [str(finalprob[x][0])+"/"+str(finalprob[x][1]) for x in finalprob]
	Dataset=[Key] + Dataset	
	outputwriter=csv.writer(open('Asymmetricdata1.csv', 'wb'))
	for row in Dataset:
		outputwriter.writerow(row)
		
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
			if random.random() <.5: 
				row[3]=1
			else:
				row[3]=0
		else:
			if random.random() <.9: 
				row[3]=0
			else:
				row[3]=1
		if random.random() < .99: row[4]=1
		else: row[4]=0
		if row[0]==0 : 
			if random.random() <.1:
				row[4]=0
	outputwriter=csv.writer(open('Asymmetricdata2.csv', 'wb'))
	for row in testdata:
		outputwriter.writerow(row)
	return testdata

#depcomposes 1 var into 2 vars
def decompose_var(columns,data):
	decomposeddata=[]
	for i in xrange(len(data)):
		decomposedrow=[]
		for column in columns:
			Var=data[i][column]
#			if i ==0:
#				print Var
##				returnvar=[Var+'a',Var+'b']			
			if Var == 1:
				if random.random() > .5: returnvar=[0,1]
				else: returnvar=[1,0]
			else:
				if random.random() > .5: returnvar=[1,1]
				else: returnvar=[0,0]
			decomposedrow=decomposedrow+returnvar
		decomposeddata.append(decomposedrow)
	return decomposeddata
def asymmetricDatawithInteraction2(obs):	
	Key=[[.9,.9],[.97,.15],[.9,.5],[.15,.97],[.5,.9],[.5,.5],[.5,.5]]

	Initial_Conditions=[0]*int(obs/2) +[1]*int(obs/2)

	Dataset = [
	    [i] + [
	        i if random.random() < k[i] else 1 - i
	        for k in Key
	    ]
	    for i in Initial_Conditions
	]
	decomposecolumns=[1,2,3,4,5,6,7]
	Dataset=decompose_var(decomposecolumns,Dataset)
	for i in xrange(len(Dataset)):
		Dataset[i]= [Initial_Conditions[i]] + Dataset[i]
	#printing Precision of simulated dataset.
	finalprob =dict([(i,(0,0)) for i in xrange(len(Dataset[0]))])
	for i in finalprob.keys():
		if i == 0: continue
		ColumnsProb=[0.0,0.0,0.0,0.0]
		for row in Dataset:
			if row[i]==0 and row[0]==0: ColumnsProb[0]=ColumnsProb[0]+1
			if row[i]==1 and row[0]==0: ColumnsProb[1]=ColumnsProb[1]+1
			if row[i]==0 and row[0]==1: ColumnsProb[2]=ColumnsProb[2]+1
			if row[i]==1 and row[0]==1: ColumnsProb[3]=ColumnsProb[3]+1
		finalprob[i]=(round(ColumnsProb[3]/(ColumnsProb[3]+ColumnsProb[1]),2),round(ColumnsProb[0]/(ColumnsProb[0]+ColumnsProb[2]),2))
		
	#Getting probability
	Key= [str(finalprob[x][0])+"/"+str(finalprob[x][1]) for x in finalprob]
	Dataset=[Key] + Dataset	
	outputwriter=csv.writer(open('Asymmetricdata2.csv', 'wb'))
	for row in Dataset:
		outputwriter.writerow(row)
	print Key
	return Dataset

if __name__=="__main__":	
		

	#createAsymmetricdata(5000)
	
	#code to verify asymmetric data with interactions
	createstratumdata2(1000)
#	confusionMatrix(map(operator.itemgetter(1), testdata) ,map(operator.itemgetter(0), testdata),0)
#	confusionMatrix(map(operator.itemgetter(2), testdata) ,map(operator.itemgetter(0), testdata),0)
#	confusionMatrix(map(operator.itemgetter(3), testdata) ,map(operator.itemgetter(0), testdata),0)
#	confusionMatrix(map(operator.itemgetter(4), testdata) ,map(operator.itemgetter(0), testdata),0)
#	for row in testdata:
#		if row[1]==1 and row[2]==1:
#			row[3]=1
#		else:
#			row[3]=0
#	confusionMatrix(map(operator.itemgetter(3), testdata) ,map(operator.itemgetter(0), testdata),0)
#	
#	testdata=asymmetricDatawithInteraction2(1000)
#	for i in xrange(len(testdata)):
#		if i ==0: testdata[i].append("merged")
#		elif (testdata[i][2]==1 and testdata[i][3]==0) or (testdata[i][2]==0 and testdata[i][3]==1) :
#			testdata[i].append(1)
#		else:
#			testdata[i].append(0)

		
	

