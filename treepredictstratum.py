import math, random
class decisionnode:
	def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
		self.col=col
		self.value=value
		self.results=results
		self.tb=tb
		self.fb=fb

# Divides a set on a specific column. Can handle numeric
# or nominal values
def divideset(rows,column,value):
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


# Create counts of possible results based on the corresponding response variable
def uniquecounts(rows,colindex):
   results={}
   for row in rows:
      # The result is the last column
      r=row[colindex]
      if r not in results: results[r]=0
      results[r]+=1
   return results

def finaluniquecounts(rows,purevar):
	colindex=len(rows[0])/2 +purevar 
	results={}
	for row in rows:
		r=row[colindex]
		if r not in results: results[r]=0
		results[r]+=1
#	results=[purevar,results] #makes the return variable, classification
	return [purevar, results]

# Entropy is the sum of p(x)log(p(x)) across all 
# the different possible results
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


def getwidth(tree):
  if tree.tb==None and tree.fb==None: return 1
  return getwidth(tree.tb)+getwidth(tree.fb)

def getdepth(tree):
  if tree.tb==None and tree.fb==None: return 0
  return max(getdepth(tree.tb),getdepth(tree.fb))+1


from PIL import Image,ImageDraw

def drawtree(tree,jpeg='tree.jpg'):
  w=getwidth(tree)*100
  h=getdepth(tree)*100+120

  img=Image.new('RGB',(w,h),(255,255,255))
  draw=ImageDraw.Draw(img)

  drawnode(draw,tree,w/2,20)
  img.save(jpeg,'JPEG')
  
def drawnode(draw,tree,x,y):
  if tree.results==None:
    # Get the width of each branch
    w1=getwidth(tree.fb)*100
    w2=getwidth(tree.tb)*100

    # Determine the total space required by this node
    left=x-(w1+w2)/2
    right=x+(w1+w2)/2

    # Draw the condition string
    draw.text((x-20,y-10),str(tree.col)+':'+str(tree.value),(0,0,0))

    # Draw links to the branches
    draw.line((x,y,left+w1/2,y+100),fill=(255,0,0))
    draw.line((x,y,right-w2/2,y+100),fill=(255,0,0))
    
    # Draw the branch nodes
    drawnode(draw,tree.fb,left+w1/2,y+100)
    drawnode(draw,tree.tb,right-w2/2,y+100)
  else:
    txt=' \n'.join(['%s:%d'%v for v in tree.results.items()])
    draw.text((x-20,y),txt,(0,0,0))


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
#Creates a boostrap sample of the rows
def sample_wr(population, k):
	"Chooses k random elements (with replacement) from a population"
	n = len(population)
	_random, _int = random.random, int  # speed hack
	return [ population[j] for j in [_int(_random() * n) for i in xrange(k)]]

def randomForest(data,trees_number):
#	forest=map(lambda x : buildtree(sample_wr(data,len(data))), xrange(trees_number) )
	forest=[buildtree(sample_wr(data,len(data))) for i in xrange(trees_number)] #builds a list of trees
	#classifiedvalues=map(classifyrows,[data]*len(forest),forest) #classifies data with list of trees
	#classifiedvalues=map(uniquecounts,zip(*classifiedvalues))
	#classifiedvalues=map(getmax,classifiedvalues)
	#calcerror(classifiedvalues,data)
	row_count=len(data)
	SimilarityMatrix=[[ reduce(lambda a, b: a + b, map(lambda x: proximity(data[i],data[j],x),forest)) for i in xrange(j+1)] for j in xrange(row_count)]
	return SimilarityMatrix


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
		
		
		
		
#Creation of clusters
class DistanceMatrix:
	def __init__(self, matrix):
		self.matrix=matrix
	def getclusterdistance(self, cluster1, cluster2): #inputs are lists
		dist=0
		divisor= 1.0/(len(cluster1) * len(cluster2))
		for i in xrange(len(cluster1)):
			for j in xrange(len(cluster2)):
				dist+=self.getrowdistance(cluster1[i],cluster2[j])/divisor
		return dist
	
	def getrowdistance(self, row1, row2):
		if row1 > row2:
			return self.matrix[row1][row2]
		else:
			return self.matrix[row2][row1]
		
class bicluster:
	def __init__(self,members,left=None,right=None,distance=None,id=None):
		self.members=members
		self.left=left
		self.right=right
		self.id=id
		self.distance=distance

def hcluster(rows):
	currentclustid=-1

	 # Clusters are initially just the rows
	clust=[bicluster([i],id=i) for i in xrange(len(rows))]
	CurrentMatrix=DistanceMatrix(rows)
	distances={}
	while len(clust)>1:
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
		print clust[lowestpair[0]].id, clust[lowestpair[1]].id
		# create the new cluster
		newcluster=bicluster(mergevec, left=clust[lowestpair[0]], right=clust[lowestpair[1]], distance=closest, id=currentclustid)
		
		# cluster ids that weren't in the original set are negative
		currentclustid-=1
		del clust[lowestpair[1]]
		del clust[lowestpair[0]]
		clust.append(newcluster)
	for i in  distances: print i, distances[i]
	return clust[0]
		
def printclust(clust,labels=None,n=0):
	# indent to make a hierarchy layout
	for i in range(n): print ' ',
	if clust.id<0:
	 	# negative id means that this is branch
		print '-'
	else:
	  	# positive id means that this is an endpoint
		if labels==None: print clust.id
		else: print labels[clust.id]

	# now print the right and left branches
	if clust.left!=None: printclust(clust.left,labels=labels,n=n+1)
	if clust.right!=None: printclust(clust.right,labels=labels,n=n+1)