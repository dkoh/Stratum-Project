#!/usr/bin/env python
# encoding: utf-8
"""
TestAsym.py

Created by Derek Koh on 2011-07-19.

"""
import Forest1,random, csv

inputdict2={'readDataset':("Asymmetricdata.csv",0,0),'number_of_trees':10, 'scoref':2, 'p_type':1}




random.seed(1234567890)

#number_of_trees=[20]
#scoref=[1]
#p_type=[1]
#finaloutput=[]
#datasets=["Asymmetricdata1.csv","Asymmetricdata2.csv"]
#for dataset in datasets:
#    for i in scoref: 
#        for j in p_type:

#            for k in number_of_trees:
#                inputdict={'readDataset':(dataset,0,0),'number_of_trees':k, 'scoref':i, 'p_type':j}
#                Newforest=Forest1.AsymForestVarSelection(**inputdict)
#                finaloutput.append([dataset,i,j,k]+list(zip(*Newforest.ranks)[0]))
#                percentile=list(zip(*Newforest.ranks)[1])
#                finaloutput.append([dataset,i,j,k]+[round(rankindex/percentile[0],2) for rankindex in percentile])
#  
#outputwriter=csv.writer(open('AsymmetricForestSimulation.csv', 'wb'))
#for row in finaloutput:

#inputdict={'readDataset':("Asymmetricdata1.csv",0,0),'number_of_trees':20, 'scoref':10, 'p_type':10}

iterations=2
topdata, features=Forest1.read_data("Asymmetricdata1.csv",0,0)
statoutput=[]
for i in xrange(iterations):
    data=random.sample(topdata,1000)
    inputdict={'data':data,'number_of_trees':20, 'scoref':10, 'p_type':10}
    newforest=Forest1.AsymForestVarSelection(**inputdict)
    statoutput.append(newforest.both_imp_rank)
#
#input impurity: entropy, P_type: calculateP1, no_of_trees: 20
#[(0, 70.909378013964272), (4, 27.562842408223311), (16, 24.941185343158772), (26, 23.737451929332135), (25, 23.408000823851093), (9, 23.118517339358711), (10, 23.037045850660288), (6, 22.832879146958316), (23, 21.700060198765755), (14, 21.349092249118936), (3, 20.906206891420116), (21, 20.246469003146686), (2, 20.126848445334247), (5, 19.752025535320737), (28, 19.616601754594843), (24, 19.163165400854993), (11, 18.539137774846864), (17, 17.900999032698127), (27, 17.816452308532071), (8, 17.618345041818621), (18, 16.261446897350073), (15, 16.015427151121688), (19, 15.595840593165379), (20, 15.539281274948635), (29, 15.354867996923277), (13, 14.304558450042292), (12, 14.191899892732939), (22, 13.577675475542577), (1, 8.7186494043139078), (7, 4.8335916794523586)]
#input impurity: entropy, P_type: calculateP1, no_of_trees: 20
#[(0, 58.213243692491112), (15, 26.998914617424351), (8, 23.630217677314398), (27, 22.22497945592368), (20, 21.017038065219804), (2, 20.939968855268177), (23, 20.544305280763659), (25, 20.528602285426246), (9, 20.416500461281831), (6, 20.361399539745381), (21, 19.794739406731701), (28, 19.619757627119981), (12, 19.564657895828404), (4, 19.161041595690577), (13, 18.563130218868899), (22, 18.493452089673607), (24, 18.366440156366561), (26, 17.594365587950406), (5, 17.51037045553263), (11, 17.343965142406681), (16, 16.183120722199071), (17, 15.000175385199974), (19, 14.947978756922506), (3, 14.66885131796116), (14, 14.26532830393505), (18, 13.752937768590076), (10, 13.10134527125866), (29, 11.235563441934136), (7, 10.253878000894515), (1, 4.9781639506038964)]

    
#get Average percentile
avgdict={}
for i in xrange(iterations):
    maxkey=max(statoutput[i][0], key=statoutput[i].get)-1
    for key in statoutput[i]:
        if i ==0: avgdict[key]=[0,0]
        avgdict[key][0]+=float(statoutput[i][key][0])/statoutput[i][maxkey][0]
        avgdict[key][1]+=float(statoutput[i][key][1])
        if i == iterations-1:
            avgdict[key][0]=avgdict[key][0]/iterations   
            avgdict[key][1]=avgdict[key][1]/iterations     

    




     

