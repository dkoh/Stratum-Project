#!/usr/bin/env python
# encoding: utf-8
"""
TestAsym.py

Created by Derek Koh on 2011-07-19.

"""
import Forest1,random, csv,time

#inputdict2={'readDataset':("Asymmetricdata.csv",0,0),'number_of_trees':10, 'scoref':2, 'p_type':1}

# START OF MAIN CODE
#random.seed(999999)
#number_of_trees=[50,100,200]
#scoref=[2,3]
#p_type=[1,2,3]
#finaloutput_pct=[]
#finaloutput_rank=[]
#datasets=["realdata3pct.csv"]
#iterations=10
#
#start_time=time.time()
#for dataset in datasets:
#    topdata, features=Forest1.read_data(dataset,0)
#    no_of_columns=len(topdata[0])-1
#    for i in scoref: 
#        for j in p_type:
#            for k in number_of_trees:
#                #do replicates
#                for l in xrange(iterations):
#                    iter_time=time.time()
#                    data=random.sample(topdata,1000)
#                    inputdict={'data':data,'number_of_trees':k, 'scoref':i, 'p_type':j, 'start':1}
#                    newforest=Forest1.AsymForestVarSelection(**inputdict)
#                    #statoutput.append(newforest.both_imp_rank) 
#                    statoutput=[-99]*no_of_columns
#                    for key in newforest.both_imp_rank:
#                        statoutput[key]=newforest.both_imp_rank[key][1]
#                    finaloutput_pct.append([dataset,i,j,k,l]+statoutput + [time.time()-iter_time])       
#outputwriter=csv.writer(open('realdataresults2.csv', 'wb'))
#for row in finaloutput_pct:
#    outputwriter.writerow(row)
#print "Completed. Total Time: ", time.time()-start_time

# END OF MAIN CODE


#def printshit(shit):
#    positive=0.0
#    positivecount=0
#    negative=0.0
#    negativecount=0
#    for i in xrange(6):
#        if i+1 in shit: 
#            positive+=shit[i+1][1]
#            positivecount+=1
#        if i+7 in shit: 
#            negative+=shit[i+7][1]
#            negativecount+=1
#    print positive/positivecount, negative/negativecount
#    
#topdata, features=Forest1.read_data("Asymmetricdata1.csv",0,0)
#iterations=1
#for i in xrange(iterations):
#    data=random.sample(topdata,500)
#    inputdict={'data':data,'number_of_trees':50, 'scoref':1, 'p_type':1}
#    newforest=Forest1.AsymForestVarSelection(**inputdict)
#    print newforest.both_imp_rank[1], newforest.both_imp_rank[7]
    
 
def getProportions(set1, set2):
    if len(set1[0])!=len(set2[0]): print "error error"
    responseidx=len(set1[0])-1
    set1pro=[1 if set1[i][responseidx]==1 else 0 for i in xrange(len(set1))]
    set2pro=[1 if set2[i][responseidx]==1 else 0 for i in xrange(len(set2))]
    print float(sum(set1pro))/len(set1pro),float(sum(set2pro))/len(set2pro),float(sum(set1pro+set2pro))/len(set1pro+set2pro)

    
   
#TEST OF THE scoref FUNCTIONS
random.seed(999999)
topdata, features=Forest1.read_data("asymmetricdata1.csv",0,0)
data=random.sample(topdata,1000)
inputdict={'data':data,'number_of_trees':1, 'scoref':1, 'p_type':1, 'start':1}
newforest=Forest1.AsymForestVarSelection(**inputdict)
newforest.printtree(newforest.classifier[0])
 
# newforest.classifier[0].puritychg

#summarydata=[]

#for repeat1 in xrange(10):
#    data=random.sample(topdata,3000)
#    inputdict={'data':data,'number_of_trees':10, 'scoref':3, 'p_type':1}
#    newforest=Forest1.AsymForestVarSelection(**inputdict)
#    col=3
#    value=1
#    (set1,set2)=newforest.divideset(newforest.data,col,value) 
#    (set3,set4)=newforest.divideset(newforest.data,col+6,value) 
##    getProportions(set1,set2)
#    print newforest.informationGain(.5,set1,set2,newforest.data),newforest.informationGain(.5,set3,set4,newforest.data)
#    summarydata.append( 1 if newforest.informationGain(.5,set1,set2,newforest.data)>newforest.informationGain(.5,set3,set4,newforest.data) else 0)
#print float(sum(summarydata)) / len(summarydata)




    
        

#    counts=[Forest1.uniquecounts(set1),Forest1.uniquecounts(set2)]
#    print value, round(float(counts[0][1])/(counts[0][1]+counts[0][-1]),3), round(float(counts[1][1])/(counts[1][1]+counts[1][-1]),3)#sorted(counts[0], key=counts[0].__getitem__, reverse=True)[0],sorted(counts[1], key=counts[1].__getitem__, reverse=True)[0]
#print Forest1.uniquecounts(newforest.data)
#    outputwriter=csv.writer(open('subsetissue.csv', 'wb'))
#    for row in newforest.data:
#        outputwriter.writerow([row[1],row[7], row[len(row)-1]])

#EMPIRICAL RESEARCH PORTION
#0.04 0.0145454545455 0.0408695652174 0.14
#data, features=Forest1.read_data("subsetissue.csv",0)
#inputdict={'data':data,'number_of_trees':10, 'scoref':1, 'p_type':1}
#newforest=Forest1.AsymForestVarSelection(**inputdict) 
#(set1,set2)=newforest.divideset(newforest.data,0,1) 
#newforest.informationGain(.5,set1,set2,newforest.data)
#print newforest.asymginiimpurity1(set1), newforest.asymginiimpurity1(set2)



#
#topdata, features=Forest1.read_data("realdata.csv",0,-99)
#data=random.sample(topdata,1000)
#inputdict={'data':data,'number_of_trees':50, 'scoref':2, 'p_type':4,'start':1}
#newforest=Forest1.AsymForestVarSelection(**inputdict) 
#print newforest.both_imp_rank
