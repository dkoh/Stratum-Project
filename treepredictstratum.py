import math, random
class decisionnode:
	def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
		self.col=col
		self.value=value
		self.results=results
		self.tb=tb
		self.fb=fb

# Divides a set on a specific column.
def divideset(rows,column,value):
   # Make a function that tells us if a row is in the first group (true) or the second group (false)
   split_function=None
   if isinstance(value,int) or isinstance(value,float):
      split_function=lambda row:row[column]>=value
   else:
      split_function=lambda row:row[column]==value   
   # Divide the rows into two sets and return them
   set1=[row for row in rows if split_function(row)]
   set2=[row for row in rows if not split_function(row)]
   return (set1,set2)

# Create counts of possible results based on the corresponding response variable
def uniquecounts(rows,colindex):
   results={}
   for row in rows:
      # The result is the last column
      r=row[colindex]
      if r not in results: results[r]=0
      results[r]+=1
   return results

#This is only used for the leaf node to return the x used and the freq of the response
def finaluniquecounts(rows,purevar):
	colindex=len(rows[0])/2 +purevar 
	results={}
	for row in rows:
		r=row[colindex]
		if r not in results: results[r]=0
		results[r]+=1
	return [purevar, results]

# Entropy is the sum of p(x)log(p(x)) 
def entropy(rows, colindex):
   from math import log
   log2=lambda x:log(x)/log(2)  
   results=uniquecounts(rows, colindex)
   # Now calculate the entropy
   ent=0.0
   for r in results.keys():
      p=float(results[r])/len(rows)
      ent=ent-p*log2(p)
   return ent

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

#Classifies the Tree
def classify(observation,tree):
  if tree.results!=None:
    return tree.results
  else:
    v=observation[tree.col]
    branch=None
    if isinstance(v,int) or isinstance(v,float):
      if v>=tree.value: branch=tree.tb
      else: branch=tree.fb
    else:
      if v==tree.value: branch=tree.tb
      else: branch=tree.fb
    return classify(observation,branch)

#Returns 1 if 2 rows are in the same leaf in the tree and 0 o/w
def proximity(obs1,obs2,tree):
	if tree.results!=None:
		return 1
	else:
		v=obs1[tree.col]
		w=obs2[tree.col]
		branch=None
		if isinstance(v,int) or isinstance(v,float):
			if v>=tree.value and w >=tree.value: branch=tree.tb
			elif v < tree.value and w < tree.value: branch=tree.fb
		else:
			if v==tree.value and w==tree.value: branch=tree.tb
			elif v!=tree.value and w!=tree.value: branch=tree.fb
		if branch ==None: return 0
		else: return proximity(obs1,obs2,branch)

#Creates a boostrap sample of the rows
def sample_wr(population, k):
	"Chooses k random elements (with replacement) from a population"
	n = len(population)
	_random, _int = random.random, int  # speed hack
	return [ population[j] for j in [_int(_random() * n) for i in xrange(k)]]

#main function for stratum forest
def stratumForest(data,trees_number):
	forest=[buildtree(sample_wr(data,len(data))) for i in xrange(trees_number)] #builds a list of trees
	row_count=len(data)
	SimilarityMatrix=[[ reduce(lambda a, b: a + b, map(lambda x: proximity(data[i],data[j],x),forest)) for i in xrange(j+1)] for j in xrange(row_count)]
 	clusters= hcluster(SimilarityMatrix,10) #Start clustering agorithm
	return reduce(lambda a, b: dict(a.items() + b.items()), [{i:clusters[i].members} for i in xrange(len(clusters))])
	


def buildForest(data):# will need to use this in the future to get OOB
	rows=sample_wr(data, len(data))	
	#oob=[ i for i in data if i not in rows]
	return buildtree(rows)

#takes a list with the len/2 predictors and the other half of the dataset is the response
def buildtree(rows,mtry=None):
	if len(rows)==0: return decisionnode()

	# Set up some variables to track the best criteria
	scoref=entropy
	best_gain=0.0
	best_criteria=None
	best_sets=None
  	min_var=-5
	min_col_value=9999999999
	column_count=len(rows[0])/2
	#Get a Random Subset
	if mtry==None: 
		mtry1=int(math.sqrt(column_count-1)) 
		columns=random.sample(range(column_count),mtry1)
	else:  
		columns=range(column_count)
	for col in columns:
		#find column score for each row
		predictorindex=column_count+col
		current_score=scoref(rows,predictorindex)
		if current_score == 0: 
			min_var=col
			min_col_value=current_score
			break
		elif current_score < min_col_value:
			min_var=col
			min_col_value=current_score
		# Generate the list of different values in this column
		column_values={}
		for row in rows:
			column_values[row[col]]=1
		# Now try dividing the rows up for each value
		# in this column
		for value in column_values.keys():
			(set1,set2)=divideset(rows,col,value)
      
		# Information gain
		p=float(len(set1))/len(rows)
		gain=current_score-p*scoref(set1,predictorindex)-(1-p)*scoref(set2,predictorindex)
		if gain>best_gain and len(set1)>0 and len(set2)>0:
			best_gain=gain
			best_criteria=(col,value)
			best_sets=(set1,set2)
  # Create the sub branches   
	if best_gain>0:
		trueBranch=buildtree(best_sets[0],mtry=mtry)
		falseBranch=buildtree(best_sets[1],mtry=mtry)
		return decisionnode(col=best_criteria[0],value=best_criteria[1],tb=trueBranch,fb=falseBranch)
	else:
		return decisionnode(results=finaluniquecounts(rows,min_var))
		
#Loads the main distance matrix
class DistanceMatrix:
	def __init__(self, matrix):
		self.matrix=matrix
	def getclusterdistance(self, cluster1, cluster2): #inputs are lists
		dist=0
		divisor= (len(cluster1) * len(cluster2))
		for i in xrange(len(cluster1)):
			for j in xrange(len(cluster2)):
				dist+=self.getrowdistance(cluster1[i],cluster2[j])/divisor
		return dist
	
	def getrowdistance(self, row1, row2):
		if row1 > row2:
			return self.matrix[row1][row2]
		else:
			return self.matrix[row2][row1]
	
#Class for each node of the cluster	
class bicluster:
	def __init__(self,members,left=None,right=None,distance=None,id=None):
		self.members=members
		self.left=left
		self.right=right
		self.id=id
		self.distance=distance

#Main algorithm for clustering
def hcluster(rows,clustercount):
	currentclustid=-1
	
	 # Clusters are initially just the rows
	clust=[bicluster([i],id=i) for i in xrange(len(rows))]
	CurrentMatrix=DistanceMatrix(rows)
	distances={}
	while len(clust)>clustercount:
		lowestpair=(0,1)
		closest=CurrentMatrix.getclusterdistance(clust[0].members,clust[1].members)
		# loop through every pair looking for the smallest distance
		for i in range(len(clust)):
			for j in range(i+1,len(clust)):
				# distances is the cache of distance calculations
				if (clust[i].id,clust[j].id) not in distances: 
					distances[(clust[i].id,clust[j].id)]=CurrentMatrix.getclusterdistance(clust[i].members,clust[j].members)
				d=distances[(clust[i].id,clust[j].id)]
				if d>closest:
					closest=d
					lowestpair=(i,j)
		# calculate the average of the two clusters
		mergevec=clust[lowestpair[0]].members
		mergevec.extend(clust[lowestpair[1]].members)
		# create the new cluster
		newcluster=bicluster(mergevec, left=clust[lowestpair[0]], right=clust[lowestpair[1]], distance=closest, id=currentclustid)
		
		# cluster ids that weren't in the original set are negative
		currentclustid-=1
		del clust[lowestpair[1]]
		del clust[lowestpair[0]]
		clust.append(newcluster)
#	for i in  distances: print i, distances[i]
	return clust


def logisticregressionR(data):
	data1=zip(*data)
	features=['col{0}'.format(i) for i in xrange(len(data[0]))]
	columns=[robjects.FloatVector(col) for col in data1]
	Rdata = robjects.r['data.frame'](**dict(zip(features,columns)))
	Rformula =  robjects.r['as.formula']('{0} ~ {1} -1'.format(features[-1],reduce(lambda x,y: x + '+' +  y, features[:-1] )))
	rpart_params = {'formula' : Rformula, 'data' : Rdata, 'family' : "binomial"}
	model=robjects.r.glm(**rpart_params)
	return (model[9],model[10])
	