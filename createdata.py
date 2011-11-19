#!/usr/bin/env python
# encoding: utf-8
"""
createdata.py

Created by Derek Koh on 2011-07-10.
Copyright (c) 2011 __MyCompanyName__. All rights reserved.
"""

import random, csv
random.seed(1234567890)

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

def createAsymmetricdata(obs,numberofpredictors):	
	obs=1000
	Key= [[max(.5,random.random()), max(.5,random.random())] for i in xrange(numberofpredictors)]

	Initial_Conditions=[0]*int(obs/2) +[1]*int(obs/2)

	Dataset = [
	    [i] + [
	        i if random.random() < k[i] else 1 - i
	        for k in Key
	    ]
	    for i in Initial_Conditions
	]

	Key= [[round(y,2) for y in x] for x in Key]
	Key= zip(*Key) 
	Dataset=[[0]+list(Key[0]),[1]+list(Key[1])]+ Dataset	
	outputwriter=csv.writer(open('Asymmetricdata.csv', 'wb'))
	for row in Dataset:
		outputwriter.writerow(row)
createAsymmetricdata(1000,50)
