import math, random 
class DecisionNode:
	def __init__(self,col=-1,value=None,results=None,tb=None,fb=None,puritychg={}):
		self.col=col
		self.value=value
		self.results=results
		self.tb=tb
		self.fb=fb
		self.puritychg=puritychg


class MachineLearningObject(object):
	def __init__(self, *args, **kwargs):
		if 'readDataset' in kwargs:
			self.data, self.features = self.read_data(*kwargs['readDataset'])
	# Entropy is the sum of p(x)log(p(x)) across all 
	# the different possible results
	def entropy(self,rows):
		 from math import log
		 log2=lambda x:log(x)/log(2)  
		 results=uniquecounts(rows)
		 # Now calculate the entropy
		 ent=0.0
		 for r in results.keys():
				p=float(results[r])/len(rows)
				ent=ent-p*log2(p)
		 return ent

			
	#Reads csv data in 
	def read_data(self,filename,stringonly=0,dependantvar=-99):
		import csv
		data = list(csv.reader(open(filename, "rb")))
		if dependantvar != -99:
			for row in data:
				row.append(row.pop(dependantvar))
		featureNames = data[0]
		featureNames= dict(zip(range(len(featureNames)), featureNames))
		data = data[1:]
		if stringonly==0:
			columns= [x for x in xrange(len(data[0]))  if is_number(data[0][x])]
			data= tofloat(data,columns)
		return data,featureNames

class ClassificationTree(MachineLearningObject):
	def __init__(self, *args, **kwargs):
		MachineLearningObject.__init__(self,**kwargs)
		blah=self.classifyrows(self.data,self.buildtree(self.data))
		for i in blah:
			print i

	
	# Divides a set on a specific column. Can handle numeric
	# or nominal values
	def divideset(self,rows,column,value):
		 # Make a function that tells us if a row is in 
		 # the first group (true) or the second group (false)
		 split_function=None
		 if isinstance(value,int) or isinstance(value,float):
				split_function=lambda row:row[column]>=value
		 else:
				split_function=lambda row:row[column]==value

		 # Divide the rows into two sets and return them
		 set1=[row for row in rows if split_function(row)]
		 set2=[row for row in rows if not split_function(row)]
		 return (set1,set2)
		
	def classifyrows(self,rows,tree):
		return map(self.classify, rows,[tree]*len(rows))

	def classify(self,observation,tree):
		if tree.results!=None:
			return tree.results.keys()
		else:
			v=observation[tree.col]
			branch=None
			if isinstance(v,int) or isinstance(v,float):
				if v>=tree.value: branch=tree.tb
				else: branch=tree.fb
			else:
				if v==tree.value: branch=tree.tb
				else: branch=tree.fb
			return self.classify(observation,branch)


	def buildtree(self,rows,scoref=None,randomcolumns=1):
		if scoref==None: scoref=self.entropy
		if len(rows)==0: return decisionnode()
		current_score=scoref(rows)
		# Set up some variables to track the best criteria
		best_gain=0.0
		best_criteria=None
		best_sets=None

		#The following statement sets columns to either be random for used by random forest or uses all columns
		if randomcolumns==1:
			mtry=int(math.sqrt(len(rows[0])-1))
			columns=random.sample(range(len(rows[0])-1),mtry)
		else:
			columns=range(len(rows[0])-1)	
		for col in columns:
		  # Generate the list of different values in
		  # this column
		  column_values={}
		  for row in rows:
			column_values[row[col]]=1
			# Now try dividing the rows up for each value
			# in this column
		  for value in column_values.keys():
			(set1,set2)=self.divideset(rows,col,value)
			
		    # Information gain
			p=float(len(set1))/len(rows)
			gain=current_score-p*scoref(set1)-(1-p)*scoref(set2)
			if gain>best_gain and len(set1)>0 and len(set2)>0:
				best_gain=gain
				best_criteria=(col,value)
				best_sets=(set1,set2)
		# Create the sub branches   
		if best_gain>0:
			trueBranch=self.buildtree(best_sets[0])
			falseBranch=self.buildtree(best_sets[1])
			return DecisionNode(col=best_criteria[0],value=best_criteria[1],
			                    tb=trueBranch,fb=falseBranch)
		else:
		  return DecisionNode(results=uniquecounts(rows))

def is_number(s):
		try:
				float(s)
				return True
		except ValueError:
				return False

#converts columns to floats
def tofloat(data1,columns):
	for row in data1:
		for column in columns:
			row[column]= float(row[column])
	return data1

#Creates a boostrap sample of the rows
def sample_wr(population, k):
	"Chooses k random elements (with replacement) from a population"
	n = len(population)
	_random, _int = random.random, int  # speed hack
	return [ population[j] for j in [_int(_random() * n) for i in xrange(k)]]




	

# Create counts of possible results (the last column of 
# each row is the result)
def uniquecounts(rows):
	 results={}
	 for row in rows:
			# The result is the last column
			r=row[len(row)-1]
			if r not in results: results[r]=0
			results[r]+=1
	 return results

# Probability that a randomly placed item will
# be in the wrong category
def giniimpurity(rows, oneside=0):
	bias=1
	total=len(rows)
	if total <=1: return 0
	else:
		counts=uniquecounts(rows)
		if oneside==1 and sorted(counts, key=counts.__getitem__, reverse=True)[0]!= bias:
			#misclassindex= 1 - (float(counts[1])/total)
			return -99
			# gini=0.0
			# for k1 in counts:
			# 	gini+=(float(counts[k1])/total)**2
			# return misclassindex-(1-gini-misclassindex)
		else:
			gini=0.0
			for k1 in counts:
				gini+=(float(counts[k1])/total)**2
			return 1-gini






def printtree(tree,indent=''):
	 # Is this a leaf node?
	 if tree.results!=None:
			print str(tree.results)
	 else:
			# Print the criteria
			print str(tree.col)+':'+str(tree.value)+'? '

			# Print the branches
			print indent+'T->',
			printtree(tree.tb,indent+'  ')
			print indent+'F->',
			printtree(tree.fb,indent+'  ')


def getwidth(tree):
	if tree.tb==None and tree.fb==None: return 1
	return getwidth(tree.tb)+getwidth(tree.fb)

def getdepth(tree):
	if tree.tb==None and tree.fb==None: return 0
	return max(getdepth(tree.tb),getdepth(tree.fb))+1

# 
# def classifyrows(rows,tree):
# 	return map(classify, rows,[tree]*len(rows))
# 	
# def classify(observation,tree):
# 	if tree.results!=None:
# 		return tree.results.keys()
# 	else:
# 		v=observation[tree.col]
# 		branch=None
# 		if isinstance(v,int) or isinstance(v,float):
# 			if v>=tree.value: branch=tree.tb
# 			else: branch=tree.fb
# 		else:
# 			if v==tree.value: branch=tree.tb
# 			else: branch=tree.fb
# 		return classify(observation,branch)

def prune(tree,mingain):
	# If the branches aren't leaves, then prune them
	if tree.tb.results==None:
		prune(tree.tb,mingain)
	if tree.fb.results==None:
		prune(tree.fb,mingain)
		
	# If both the subbranches are now leaves, see if they
	# should merged
	if tree.tb.results!=None and tree.fb.results!=None:
		# Build a combined dataset
		tb,fb=[],[]
		for v,c in tree.tb.results.items():
			tb+=[[v]]*c
		for v,c in tree.fb.results.items():
			fb+=[[v]]*c
		
		# Test the reduction in entropy
		delta=entropy(tb+fb)-(entropy(tb)+entropy(fb)/2)

		if delta<mingain:
			# Merge the branches
			tree.tb,tree.fb=None,None
			tree.results=uniquecounts(tb+fb)

def mdclassify(observation,tree):
	if tree.results!=None:
		return tree.results
	else:
		v=observation[tree.col]
		if v==None:
			tr,fr=mdclassify(observation,tree.tb),mdclassify(observation,tree.fb)
			tcount=sum(tr.values())
			fcount=sum(fr.values())
			tw=float(tcount)/(tcount+fcount)
			fw=float(fcount)/(tcount+fcount)
			result={}
			for k,v in tr.items(): result[k]=v*tw
			for k,v in fr.items(): result[k]=v*fw      
			return result
		else:
			if isinstance(v,int) or isinstance(v,float):
				if v>=tree.value: branch=tree.tb
				else: branch=tree.fb
			else:
				if v==tree.value: branch=tree.tb
				else: branch=tree.fb
			return mdclassify(observation,branch)

def variance(rows):
	if len(rows)==0: return 0
	data=[float(row[len(row)-1]) for row in rows]
	mean=sum(data)/len(data)
	variance=sum([(d-mean)**2 for d in data])/len(data)
	return variance


#Gets the key of the max values from a dictionary
def getmax(contenders):
	return max(contenders, key = lambda x: contenders.get(x) )
	
#Calc error rate
def calcerror(classifiedvalues,data):
	responsecolumn=len(data[0])-1
	correct=0
	for i in xrange(len(data)):
		if data[i][responsecolumn]== classifiedvalues[i]:
			correct=correct+1
	print correct, len(data), correct/float(len(data))


#Main code
def randomForest(data,trees_number):
	random.seed(123456789)
	forest=[buildForest(data) for i in xrange(trees_number)] #builds a list of trees
	classifiedvalues=map(classifyrows,[data]*len(forest),forest) #classifies data with list of trees
	classifiedvalues=map(uniquecounts,zip(*classifiedvalues))
	classifiedvalues=map(getmax,classifiedvalues)
	calcerror(classifiedvalues,data)
	#print reduce(lambda x, y: x+y,map(getTotalGain,forest))
	return consolidateinfoGains(map(getTotalGain,forest))
	
def consolidateinfoGains(forestpredictors):
	predictors={}
	for treepredictors in forestpredictors:
		for r in treepredictors:
			if r not in predictors: predictors[r]=0
			predictors[r]+=treepredictors[r]
	return predictors

def getTotalGain(tree):
	if tree.results !=None:
		return {}
	else:
		tb=getTotalGain(tree.tb)
		fb=getTotalGain(tree.fb)
		results= dict(tree.puritychg.items()+tb.items() + fb.items()).keys() 
		results=dict(zip(results,[0]*len(results)))
		for i in results: 
			if i in tb: results[i]+=tb[i]
			if i in fb: results[i]+=fb[i]
			if i in tree.puritychg: results[i]+=tree.puritychg[i]
		return results
	
def getwidth(tree):
	if tree.tb==None and tree.fb==None: return 1
	return getwidth(tree.tb)+getwidth(tree.fb)	
	
def buildForest(data):
	#rows=sample_wr(data, len(data))	
	#oob=[ i for i in data if i not in rows]
	return buildtree(data)
	


# def buildtree(rows,scoref=giniimpurity):
# 	if len(rows)==0: return DecisionNode()
# 	current_score=scoref(rows)
# 
# 	# Set up some variables to track the best criteria
# 	best_gain=0.0
# 	best_criteria=None
# 	best_sets=None
# 	
# 	mtry=int(math.sqrt(len(rows[0])-1))
# 	columns=random.sample(range(len(rows[0])-1),mtry)
# 
# 	for col in columns:
# 		# Generate the list of different values in
# 		# this column
# 		column_values={}
# 		for row in rows:
# 			 column_values[row[col]]=1
# 		# Now try dividing the rows up for each value
# 		# in this column
# 		for value in column_values.keys():
# 			(set1,set2)=divideset(rows,col,value)
# 			
# 			# Information gain
# 			if len(set1) > 1 and len(set2)>1:
# 				p=float(math.log(len(set1)))/float(math.log(len(set1))+math.log(len(set2)))				
# 				leftnodepurity=scoref(set1,1)
# 				if leftnodepurity == -99:
# 					leftnodepurity=current_score
# 				rightnodepurity=scoref(set2,1)
# 				if rightnodepurity==-99:
# 					rightnodepurity=current_score
# 				gain=current_score-p*leftnodepurity-(1-p)*rightnodepurity
# 				if gain>best_gain:
# 					best_gain=gain
# 					best_criteria=(col,value)
# 					best_sets=(set1,set2)
# 				
# 	# Create the sub branches   
# 	if best_gain>0:
# 		trueBranch=buildtree(best_sets[0])
# 		falseBranch=buildtree(best_sets[1])
# 		return DecisionNode(col=best_criteria[0],value=best_criteria[1],
# 												tb=trueBranch,fb=falseBranch,puritychg={best_criteria[0]:best_gain})
# 	else:
# 		return DecisionNode(results=uniquecounts(rows))

