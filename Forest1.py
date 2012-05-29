import math, random 

class DecisionNode:
	def __init__(self,col=-1,value=None,results=None,tb=None,fb=None,puritychg={}):
		self.col=col
		self.value=value
		self.results=results
		self.tb=tb
		self.fb=fb
		self.puritychg=puritychg
	def getweightedresults(self):
		return dict([(key,self.results[0]*self.results[1][key]) for key in self.results[1]])

			
		

class MachineLearningObject(object):
	def __init__(self, *args, **kwargs):
		if not hasattr(self, 'data'):
			if 'readDataset' in kwargs:
				self.data, self.features = read_data(*kwargs['readDataset'])
			elif 'data' in kwargs:
				self.data=kwargs['data']
			
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
		
	def giniimpurity(self,rows):
		total=len(rows)
		class_values=uniquecounts(rows)
		imp=0
		for k1 in class_values:
			p1=float(class_values[k1])/total
			for k2 in class_values:
				if k1==k2: continue
				p2=float(class_values[k2])/total
				imp+=p1*p2
		return imp
				

	


class Classifier(MachineLearningObject):
	def classifyData(self):
		pass
		
	def classifyInsample(self):		
		insampleClassifiedData=self.classifyData(self.data)
		dependentVarIndex=len(self.data[0])-1
		return self.calcError(insampleClassifiedData,self.data)		

	#Calc error rate
	def calcError(self,classifiedvalues,data):
		responsecolumn=len(data[0])-1
		correct=0
		for i in xrange(len(data)):
			if data[i][responsecolumn]== classifiedvalues[i]:
				correct=correct+1
		return correct, len(data), correct/float(len(data))

class Clustering(MachineLearningObject):
	pass
class VariableSelection(MachineLearningObject):
	#Sort variables from most important to least important
	def RankImportance(self):
		from operator import itemgetter
		sorted_results = sorted(self.variableimportance.iteritems(), key=itemgetter(1),reverse=True)
		justtheranks=map(itemgetter(0), sorted_results)
		return dict((justtheranks[i],i+1) for i in xrange(len(justtheranks)))
class ClassificationTree(Classifier):
	def __init__(self, *args, **kwargs):	
		if not hasattr(self, 'scoref'):
			self.scoref=self.entropy
		else: self.scoref=kwargs('scoref')
		print "Impurity Measure: {0}".format(self.scoref.__name__)
		MachineLearningObject.__init__(self,**kwargs)
		self.classifier=self.buildtree(self.data)	
	
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
		
	def classifyData(self,rows,tree=None):
		if tree == None:
			tree=self.classifier
		return map(self.classify, rows,[tree]*len(rows))

	def classify(self,observation,tree):
		if tree.results!=None:
			return tree.results.keys()[0]
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
			
	def printtree(self,tree=None,indent=''):
		if tree==None: tree=self.classifier
		# Is this a leaf node?
		if tree.results!=None:
			print str(tree.results)
		else:
			# Print the criteria
			print str(tree.col)+':'+str(tree.value)+'? '
			# Print the branches
			print indent+'T->',
			self.printtree(tree.tb,indent+'  ')
			print indent+'F->',
			self.printtree(tree.fb,indent+'  ')
	
	#print information gain for each node
	def printtree2a(self,tree=None):
		col=1
		if tree==None: tree=self.classifier[0]
		if col in tree.puritychg or col+6 in tree.puritychg: print tree.puritychg
		if tree.results==None:
			# Print the criteria
			if col in tree.puritychg or col+6 in tree.puritychg: print tree.puritychg
			# Print the branches
			self.printtree2(tree.fb)
			self.printtree2(tree.tb)

	def informationGain(self,current_score,set1,set2,rows):
		p=float(len(set1))/len(rows)
		gain=0
		if p != 0 and p != 1:				
			gain=current_score-p*self.scoref(set1)-(1-p)*self.scoref(set2)
		return gain
			
	def buildtree(self,rows, randomcolumns=None):		
		if len(rows)==0: return DecisionNode()
		current_score=self.scoref(rows)
		if current_score == -99: current_score=.5
		# Set up some variables to track the best criteria
		best_gain=0.0
		best_criteria=None
		best_sets=None
		#Picks the subset of variables to find splits if randomcolumns is -1 then the splits are the sqrt of 
		#total columns.
		if randomcolumns==-1:
			mtry=int(math.sqrt(len(rows[0])-1))
			columns=random.sample(range(len(rows[0])-1),mtry)
		elif randomcolumns==None:
			columns=range(len(rows[0])-1)
		else:
			columns=random.sample(range(len(rows[0])-1),randomcolumns)	
#		if 1 in columns and 7 in columns:
#			# Generate the list of different values in
#			# this column
#			dummy_output={}
#			for col in [1,7]:
#				dummy_gain=0.0
#				column_values={}
#				for row in rows:
#					column_values[row[col]]=1
#				# Now try dividing the rows up for each value
#				# in this column
#				for value in column_values.keys():
#					(set1,set2)=self.divideset(rows,col,value)			
#					# Information gain
#					gain=self.informationGain(current_score,set1,set2,rows)	
#					if gain>dummy_gain and len(set1)>0 and len(set2)>0:
#						dummy_gain=gain
#				dummy_output[col]=dummy_gain
#			if dummy_output[1] < dummy_output[7]:
#				(set1,set2)=self.divideset(rows,1,1)	
#				self.auditsplit(current_score,set1,set2)	
#				(set1,set2)=self.divideset(rows,7,1)	
#				self.auditsplit(current_score,set1,set2)	
#				print "shit"	

		
		for col in columns:
			# Generate the list of different values in
			# this column
			column_values={}
			for row in rows:
				column_values[row[col]]=1
			# Now try dividing the rows up for each value
			# in this column
			if len(column_values)==1: continue
			for value in column_values.keys():
				(set1,set2)=self.divideset(rows,col,value)			
				# Information gain
				gain=self.informationGain(current_score,set1,set2,rows)	
				if gain>best_gain and len(set1)>0 and len(set2)>0:
					best_gain=gain
					best_criteria=(col,value)
					best_sets=(set1,set2)
		# Create the sub branches   
		if best_gain>0:
			trueBranch=self.buildtree(best_sets[0],randomcolumns)
			falseBranch=self.buildtree(best_sets[1],randomcolumns)
#			self.auditsplit(current_score,best_sets[0],best_sets[1])	
			return DecisionNode(col=best_criteria[0],value=best_criteria[1],
			                    tb=trueBranch,fb=falseBranch,puritychg={best_criteria[0]:best_gain})
		else:
			return DecisionNode(results=uniquecounts(rows))
		
	def auditsplit(self,current_score,set1,set2):
		gain=0				
		if len(set1) > 1 and len(set2)>1:
			p=self.p_type(set1,set2)	
			leftnodepurity=self.scoref(set1)
#			if leftnodepurity == -99:
#				leftnodepurity=current_score
			rightnodepurity=self.scoref(set2)
#			if rightnodepurity==-99:
#				rightnodepurity=current_score
			gain=current_score-p*leftnodepurity-(1-p)*rightnodepurity
			counts1=uniquecounts(set1)
			counts2=uniquecounts(set2)
			class1=sorted(counts1, key=counts1.__getitem__, reverse=True)[0]
			class2=sorted(counts2, key=counts2.__getitem__, reverse=True)[0]
			print uniquecounts(set1), "\t", round(gain,3), "\t", round(p,3), "\t",round(current_score,3), "\t",round(leftnodepurity,3), "\t",round(rightnodepurity,3),"\t",class1,class2
			
	
class RandomForest(ClassificationTree):
	def __init__(self,*args,**kwargs):
		if 'number_of_trees' in kwargs:
			self.number_of_trees=kwargs['number_of_trees']
		if not hasattr(self, 'scoref'):
			self.scoref=self.entropy
		MachineLearningObject.__init__(self,**kwargs)
		self.classifier=[self.buildForest(self.data) for i in xrange(self.number_of_trees)] #builds Random Forest by training a list of trees

	def classifyData(self,rows):
		classifiedvalues=map(ClassificationTree.classifyData,[self]*len(self.classifier),[rows]*len(self.classifier),self.classifier) 
		classifiedvalues=map(uniquecounts,zip(*classifiedvalues))
		return map(getmax,classifiedvalues)

	def buildForest(self, data):
		rows=sample_with_replacement(data, len(data))	
#		oob=[ i for i in data if i not in rows]
		return self.buildtree(rows,-1)

class ForestVarSelection(RandomForest,VariableSelection):
	def __init__(self,*args,**kwargs):
		self.scoref=self.entropy
		RandomForest.__init__(self,**kwargs)
		classifiedvalues=self.classifyData(self.data)
		self.variableimportance=self.consolidateinfoGains(map(self.getTotalGain,self.classifier))
		print self.RankImportance()
		
	#Consolidate information gain between trees in the forest
	def consolidateinfoGains(self,forestpredictors):
		predictors={}
		for treepredictors in forestpredictors:
			for r in treepredictors:
				if r not in predictors: predictors[r]=0
				predictors[r]+=treepredictors[r]
		return predictors
	
	#Returns a dict for the total information gain for each variable
	def getTotalGain(self, tree):
		if tree.results !=None:
			return {}
		else:
			tb=self.getTotalGain(tree.tb)
			fb=self.getTotalGain(tree.fb)
			results= dict(tree.puritychg.items()+tb.items() + fb.items()).keys() 
			results=dict(zip(results,[0]*len(results)))
			for i in results: 
				if i in tb: results[i]+=tb[i]
				if i in fb: results[i]+=fb[i]
				if i in tree.puritychg: results[i]+=tree.puritychg[i]
			return results
	
class AsymForestVarSelection(ForestVarSelection):
	def __init__(self,*args,**kwargs):
		
		#setup dictionaries for p_type and scoref
		p_type_dict={1:self.calculateP1, 2:self.calculateP2, 3:self.calculateP3}
		scoref_dict={1:self.asymginiimpurity1,2:self.asymginiimpurity2,3:self.giniimpurity}
		#Assign p_type and scoref
		self.p_type=p_type_dict.get(kwargs.get('p_type'),self.calculateP1)
		self.scoref=scoref_dict.get(kwargs.get('scoref'),self.entropy)
		print "input impurity: {0}, P_type: {1}, no_of_trees: {2}".format(self.scoref.__name__,self.p_type.__name__,kwargs['number_of_trees'])
		self.bias=1
#		#start running
		if 'start' in kwargs:
			self.start(kwargs)
		else: 
			MachineLearningObject.__init__(self,**kwargs)
	def start(self,kwargs):
		RandomForest.__init__(self,**kwargs)
		self.variableimportance=self.consolidateinfoGains(map(self.getTotalGain,self.classifier))
		self.ranks=self.RankImportance()
		self.both_imp_rank=dict((key,(self.variableimportance[key],self.ranks[key])) for key in self.variableimportance)
	def informationGain(self,current_score,set1,set2,rows):
		gain=0				
		if len(set1) > 1 and len(set2)>1:
			p=self.p_type(set1,set2)	
			leftnodepurity=self.scoref(set1)
			if leftnodepurity == -99:
				leftnodepurity=current_score
			rightnodepurity=self.scoref(set2)
			if rightnodepurity==-99:
				rightnodepurity=current_score
			gain=current_score-p*leftnodepurity-(1-p)*rightnodepurity
		return gain
	#Standard P
	def calculateP1(self, set1, set2):
		return float(len(set1))/float(len(set1)+len(set2))		
	#Ps that uses the log function
	def calculateP2(self, set1, set2):
		return float(math.log(len(set1)))/float(math.log(len(set1))+math.log(len(set2)))		
	#Ps that only consider strictly bias class
	def calculateP3(self, set1, set2):
		counts1=uniquecounts(set1)
		counts2=uniquecounts(set2)
		if sorted(counts1, key=counts1.__getitem__, reverse=True)[0]!= self.bias:
			countleft=0
		else: countleft=len(set1)
		if sorted(counts2, key=counts2.__getitem__, reverse=True)[0]!= self.bias:
			countright=0
		else: countright=len(set2)	
		if countleft+countright ==0: return 0.5
		else: return countleft/(countleft+countright)
	
	#Gini that only calculates on the bias side. Returns -99 on the complement.
	def asymginiimpurity1(self,rows):
		total=len(rows)
		if total <=1: return 0
		else:
			counts=uniquecounts(rows)
			if sorted(counts, key=counts.__getitem__, reverse=True)[0]!= self.bias:
				return -99
			else:
				gini=0.0
				for k1 in counts:
					gini+=(float(counts[k1])/total)**2
				return 1-gini
				
	# Gini is that linear on the complement and curved in the other.
	def asymginiimpurity2(self, rows):
		total=len(rows)
		if total <=1: return 0
		else:
			counts=uniquecounts(rows)
			if sorted(counts, key=counts.__getitem__, reverse=True)[0]== self.bias:
				gini=float(counts[self.bias]/total)
			else:
				gini=0.0
				for k1 in counts:
					gini+=(float(counts[k1])/total)**2
			return 1-gini

class TestTrees(AsymForestVarSelection):
	def __init__(self,*args,**kwargs):
		#setup dictionaries for p_type and scoref
		p_type_dict={1:self.calculateP1, 2:self.calculateP2, 3:self.calculateP3}
		scoref_dict={1:self.asymginiimpurity1,2:self.asymginiimpurity2,3:self.giniimpurity}
		#Assign p_type and scoref
		self.p_type=p_type_dict.get(kwargs.get('p_type'),self.calculateP1)
		self.scoref=scoref_dict.get(kwargs.get('scoref'),self.entropy)
		print "input impurity: {0}, P_type: {1}, no_of_trees: {2}".format(self.scoref.__name__,self.p_type.__name__,kwargs['number_of_trees'])
		self.bias=1
		MachineLearningObject.__init__(self,**kwargs)	
		self.printGain(self.data)
	
	def printGain(self,rows, randomcolumns=None):		
		if len(rows)==0: return DecisionNode()
		current_score=self.scoref(rows)
		if current_score == -99: current_score=.5
		
		#Picks the subset of variables to find splits if randomcolumns is -1 then the splits are the sqrt of 
		#total columns.
		if randomcolumns==-1:
			mtry=int(math.sqrt(len(rows[0])-1))
			columns=random.sample(range(len(rows[0])-1),mtry)
		elif randomcolumns==None:
			columns=range(len(rows[0])-1)
		else:
			columns=random.sample(range(len(rows[0])-1),randomcolumns)	
		columns=[1,7,25]
		for col in columns:
			best_gain=0
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
				gain=self.informationGain(current_score,set1,set2,rows)	
				if gain>best_gain and len(set1)>0 and len(set2)>0:
					best_gain=gain	
				
#			print col, best_gain
	def informationGain(self,current_score,set1,set2,rows):
		gain=0				
		if len(set1) > 1 and len(set2)>1:
			p=self.p_type(set1,set2)	
			leftnodepurity=self.scoref(set1)
			if leftnodepurity == -99:
				leftnodepurity=current_score
			rightnodepurity=self.scoref(set2)
			if rightnodepurity==-99:
				rightnodepurity=current_score
			gain=current_score-p*leftnodepurity-(1-p)*rightnodepurity
#			print "Gain \t p \t leftnode \t rightnode \t counts"
			print current_score, round(gain,3),"\t", round(p,3),"\t", round(leftnodepurity,3),"\t", round(rightnodepurity,3), uniquecounts(set1),uniquecounts(set2)
		return gain
		
class ConditionalRandomForest(RandomForest):
	def __init__(self,*args,**kwargs):
		self.scoref=self.mGini
		MachineLearningObject.__init__(self,**kwargs)
		self.createDependentvars()
		RandomForest.__init__(self,**kwargs)
		self.variableimportance=self.consolidateinfoGains(map(self.getAvgGiniforTrees,self.classifier))
		print [self.variableimportance[key]/(5.0*len(self.data)) for key in self.variableimportance]
	#Consolidate information between trees in the forest
	def consolidateinfoGains(self,forestpredictors):
		predictors={}
		for treepredictors in forestpredictors:
			for r in treepredictors:
				if r not in predictors: predictors[r]=0
				predictors[r]+=treepredictors[r]
		return predictors


	def getAvgGiniforTrees(self,tree):
		if tree.results==None:
			fb=self.getAvgGiniforTrees(tree.fb)
			tb=self.getAvgGiniforTrees(tree.tb)
			results= dict(tb.items() + fb.items()).keys() 
			results=dict(zip(results,[0]*len(results)))
			for i in results: 
				if i in tb: results[i]+=tb[i]
				if i in fb: results[i]+=fb[i]
			return results
		else:
			return tree.getweightedresults()
	def testtree(self,tree):
		if tree.results==None:
			if tree.fb==None: print "fb has none"
			else: self.testtree(tree.fb)
			if tree.tb==None: print "tb has none"
			else: self.testtree(tree.tb)
		else:
			print tree.getweightedresults().items()
		
	#create the dependant var for each predictor	
	def createDependentvars(self):
		data=self.data
		self.featurecount=len(data[0])/2
		featurecount=self.featurecount
		for i in xrange(len(data)):
			y_dep=[ 0 for k1 in xrange(featurecount)]
			for j in xrange(featurecount):
				if data[i][j+featurecount] > data[i][j]: 
					y_dep[j]=1
				else:
					y_dep[j]=-1
			data[i]=data[i]+y_dep
	#Returns the lowest gini based on all the responses
	def mGini(self,rows):
		featurecount=self.featurecount
		bestimpurity=1
		transposeddata=zip(*rows)
		for i in xrange(featurecount):
			candidateimpurity=self.giniimpurity(list(transposeddata[2*featurecount+i]))
			if candidateimpurity< bestimpurity:
				bestimpurity=candidateimpurity
		return bestimpurity
	
	#Gets the Gini for all dependent vars
	def leafGini(self,rows):
		featurecount=self.featurecount
		transposeddata=zip(*rows)
		return dict([(i,self.giniimpurity(list(transposeddata[2*featurecount+i]))) for i in xrange(featurecount)])
	
	
	def buildtree(self,rows, randomcolumns=None):		
		if len(rows)==0: return DecisionNode()
		current_score=self.scoref(rows)
		if current_score == -99: current_score=.5
		elif current_score==0: return DecisionNode(results=(len(rows),self.leafGini(rows)))
		# Set up some variables to track the best criteria
		best_gain=0.0
		best_criteria=None
		best_sets=None
		featurecount=self.featurecount
		#Picks the subset of variables to find splits if randomcolumns is -1 then the splits are the sqrt of 
		#total columns.
		if randomcolumns==-1:
			mtry=int(math.sqrt(featurecount))
			columns=random.sample(range(featurecount),mtry)
		elif randomcolumns==None:
			columns=range(featurecount)
		else:
			columns=random.sample(range(featurecount),randomcolumns)	

		for col in columns:
			# Generate the list of different values in this column
			column_values={}
			for row in rows:
				column_values[row[col]]=1
			# Now try dividing the rows up for each value in this column
			if len(column_values)==1: continue
			for value in column_values.keys():
				(set1,set2)=self.divideset(rows,col,value)
				if len(set1)==0 or len(set2)==0: continue			
				# Information gain
				gain=self.informationGain(current_score,set1,set2,rows)	
				if gain>best_gain and len(set1)>0 and len(set2)>0:
					best_gain=gain
					best_criteria=(col,value)
					best_sets=(set1,set2)
#		# Create the sub branches   
		if best_gain>0:
			trueBranch=self.buildtree(best_sets[0],randomcolumns)
			falseBranch=self.buildtree(best_sets[1],randomcolumns)	
			return DecisionNode(col=best_criteria[0],value=best_criteria[1],
			                    tb=trueBranch,fb=falseBranch,puritychg={best_criteria[0]:best_gain})
		else:
			return DecisionNode(results=(len(rows),self.leafGini(rows)))
			
####Miscellaneous Functions####
	
def read_data(filename,stringonly=0,dependantvar=-99):
#Reads csv data in. stringonly=1 means that all values will be left as string.
#otherwise  if stringonly =0 then all columns that are numbers are converted to float. 
#dependantvar specifies the dependent var column. if -99 then the last column is the dependant var
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
#Check is value is a number
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

#Tally unique values for dependent variable
def uniquecounts(rows):
	results={}
	if type(rows[0]) is not list and type(rows[0]) is not tuple :
		for r in rows:
			if r not in results: results[r]=0
			results[r]+=1			
	else:
		for row in rows:
			# The result is the last column
			r=row[len(row)-1]
			if r not in results: results[r]=0
			results[r]+=1
	return results

#Creates a bootstrap sample of the rows
def sample_with_replacement(population, k):
	"Chooses k random elements (with replacement) from a population"
	n = len(population)
	_random, _int = random.random, int  # speed hack
	return [ population[j] for j in [_int(_random() * n) for i in xrange(k)]]	

#Gets the key of the max value from a dictionary
def getmax(contenders):
	return max(contenders, key = lambda x: contenders.get(x) )


if __name__ == "__main__":
	#import TestAsym
	import teststratum