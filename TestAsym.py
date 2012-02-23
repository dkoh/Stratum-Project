#!/usr/bin/env python
# encoding: utf-8
"""
TestAsym.py

Created by Derek Koh on 2011-07-19.

"""
import Forest1,random

inputdict1={'readDataset':("Asymmetricdata1.csv",0,0),'number_of_trees':100,'scoref':'giniimpurity'}
inputdict2={'readDataset':("Asymmetricdata.csv",0,0),'number_of_trees':2, 'gini_type':1, 'p_type':1}




#where is the scoref getting initialized?!?!?!?

random.seed(1234567890)

#NewLearner1=Forest1.ForestVarSelection(**inputdict1)
NewLearner2=Forest1.RandomForest(**inputdict1)