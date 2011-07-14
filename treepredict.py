my_data=[['slashdot','USA','yes',18,'None'],
        ['google','France','yes',23,'Premium'],
        ['digg','USA','yes',24,'Basic'],
        ['kiwitobes','France','yes',23,'Basic'],
        ['google','UK','no',21,'Premium'],
        ['(direct)','New Zealand','no',12,'None'],
        ['(direct)','UK','no',21,'Basic'],
        ['google','USA','no',24,'Premium'],
        ['slashdot','France','yes',19,'None'],
        ['digg','USA','no',18,'None'],
        ['google','UK','no',18,'None'],
        ['kiwitobes','UK','no',19,'None'],
        ['digg','New Zealand','yes',12,'Basic'],
        ['slashdot','UK','no',21,'None'],
        ['google','UK','yes',18,'Basic'],
        ['kiwitobes','France','yes',19,'Basic']]

iris_data=[[5.1,3.5,1.4,0.2,'Iris-setosa'],
		[4.9,3.0,1.4,0.2,'Iris-setosa'],
		[4.7,3.2,1.3,0.2,'Iris-setosa'],
		[4.6,3.1,1.5,0.2,'Iris-setosa'],
		[5.0,3.6,1.4,0.2,'Iris-setosa'],
		[5.4,3.9,1.7,0.4,'Iris-setosa'],
		[4.6,3.4,1.4,0.3,'Iris-setosa'],
		[5.0,3.4,1.5,0.2,'Iris-setosa'],
		[4.4,2.9,1.4,0.2,'Iris-setosa'],
		[4.9,3.1,1.5,0.1,'Iris-setosa'],
		[5.4,3.7,1.5,0.2,'Iris-setosa'],
		[4.8,3.4,1.6,0.2,'Iris-setosa'],
		[4.8,3.0,1.4,0.1,'Iris-setosa'],
		[4.3,3.0,1.1,0.1,'Iris-setosa'],
		[5.8,4.0,1.2,0.2,'Iris-setosa'],
		[5.7,4.4,1.5,0.4,'Iris-setosa'],
		[5.4,3.9,1.3,0.4,'Iris-setosa'],
		[5.1,3.5,1.4,0.3,'Iris-setosa'],
		[5.7,3.8,1.7,0.3,'Iris-setosa'],
		[5.1,3.8,1.5,0.3,'Iris-setosa'],
		[5.4,3.4,1.7,0.2,'Iris-setosa'],
		[5.1,3.7,1.5,0.4,'Iris-setosa'],
		[4.6,3.6,1.0,0.2,'Iris-setosa'],
		[5.1,3.3,1.7,0.5,'Iris-setosa'],
		[4.8,3.4,1.9,0.2,'Iris-setosa'],
		[5.0,3.0,1.6,0.2,'Iris-setosa'],
		[5.0,3.4,1.6,0.4,'Iris-setosa'],
		[5.2,3.5,1.5,0.2,'Iris-setosa'],
		[5.2,3.4,1.4,0.2,'Iris-setosa'],
		[4.7,3.2,1.6,0.2,'Iris-setosa'],
		[4.8,3.1,1.6,0.2,'Iris-setosa'],
		[5.4,3.4,1.5,0.4,'Iris-setosa'],
		[5.2,4.1,1.5,0.1,'Iris-setosa'],
		[5.5,4.2,1.4,0.2,'Iris-setosa'],
		[4.9,3.1,1.5,0.1,'Iris-setosa'],
		[5.0,3.2,1.2,0.2,'Iris-setosa'],
		[5.5,3.5,1.3,0.2,'Iris-setosa'],
		[4.9,3.1,1.5,0.1,'Iris-setosa'],
		[4.4,3.0,1.3,0.2,'Iris-setosa'],
		[5.1,3.4,1.5,0.2,'Iris-setosa'],
		[5.0,3.5,1.3,0.3,'Iris-setosa'],
		[4.5,2.3,1.3,0.3,'Iris-setosa'],
		[4.4,3.2,1.3,0.2,'Iris-setosa'],
		[5.0,3.5,1.6,0.6,'Iris-setosa'],
		[5.1,3.8,1.9,0.4,'Iris-setosa'],
		[4.8,3.0,1.4,0.3,'Iris-setosa'],
		[5.1,3.8,1.6,0.2,'Iris-setosa'],
		[4.6,3.2,1.4,0.2,'Iris-setosa'],
		[5.3,3.7,1.5,0.2,'Iris-setosa'],
		[5.0,3.3,1.4,0.2,'Iris-setosa'],
		[7.0,3.2,4.7,1.4,'Iris-versicolor'],
		[6.4,3.2,4.5,1.5,'Iris-versicolor'],
		[6.9,3.1,4.9,1.5,'Iris-versicolor'],
		[5.5,2.3,4.0,1.3,'Iris-versicolor'],
		[6.5,2.8,4.6,1.5,'Iris-versicolor'],
		[5.7,2.8,4.5,1.3,'Iris-versicolor'],
		[6.3,3.3,4.7,1.6,'Iris-versicolor'],
		[4.9,2.4,3.3,1.0,'Iris-versicolor'],
		[6.6,2.9,4.6,1.3,'Iris-versicolor'],
		[5.2,2.7,3.9,1.4,'Iris-versicolor'],
		[5.0,2.0,3.5,1.0,'Iris-versicolor'],
		[5.9,3.0,4.2,1.5,'Iris-versicolor'],
		[6.0,2.2,4.0,1.0,'Iris-versicolor'],
		[6.1,2.9,4.7,1.4,'Iris-versicolor'],
		[5.6,2.9,3.6,1.3,'Iris-versicolor'],
		[6.7,3.1,4.4,1.4,'Iris-versicolor'],
		[5.6,3.0,4.5,1.5,'Iris-versicolor'],
		[5.8,2.7,4.1,1.0,'Iris-versicolor'],
		[6.2,2.2,4.5,1.5,'Iris-versicolor'],
		[5.6,2.5,3.9,1.1,'Iris-versicolor'],
		[5.9,3.2,4.8,1.8,'Iris-versicolor'],
		[6.1,2.8,4.0,1.3,'Iris-versicolor'],
		[6.3,2.5,4.9,1.5,'Iris-versicolor'],
		[6.1,2.8,4.7,1.2,'Iris-versicolor'],
		[6.4,2.9,4.3,1.3,'Iris-versicolor'],
		[6.6,3.0,4.4,1.4,'Iris-versicolor'],
		[6.8,2.8,4.8,1.4,'Iris-versicolor'],
		[6.7,3.0,5.0,1.7,'Iris-versicolor'],
		[6.0,2.9,4.5,1.5,'Iris-versicolor'],
		[5.7,2.6,3.5,1.0,'Iris-versicolor'],
		[5.5,2.4,3.8,1.1,'Iris-versicolor'],
		[5.5,2.4,3.7,1.0,'Iris-versicolor'],
		[5.8,2.7,3.9,1.2,'Iris-versicolor'],
		[6.0,2.7,5.1,1.6,'Iris-versicolor'],
		[5.4,3.0,4.5,1.5,'Iris-versicolor'],
		[6.0,3.4,4.5,1.6,'Iris-versicolor'],
		[6.7,3.1,4.7,1.5,'Iris-versicolor'],
		[6.3,2.3,4.4,1.3,'Iris-versicolor'],
		[5.6,3.0,4.1,1.3,'Iris-versicolor'],
		[5.5,2.5,4.0,1.3,'Iris-versicolor'],
		[5.5,2.6,4.4,1.2,'Iris-versicolor'],
		[6.1,3.0,4.6,1.4,'Iris-versicolor'],
		[5.8,2.6,4.0,1.2,'Iris-versicolor'],
		[5.0,2.3,3.3,1.0,'Iris-versicolor'],
		[5.6,2.7,4.2,1.3,'Iris-versicolor'],
		[5.7,3.0,4.2,1.2,'Iris-versicolor'],
		[5.7,2.9,4.2,1.3,'Iris-versicolor'],
		[6.2,2.9,4.3,1.3,'Iris-versicolor'],
		[5.1,2.5,3.0,1.1,'Iris-versicolor'],
		[5.7,2.8,4.1,1.3,'Iris-versicolor'],
		[6.3,3.3,6.0,2.5,'Iris-virginica'],
		[5.8,2.7,5.1,1.9,'Iris-virginica'],
		[7.1,3.0,5.9,2.1,'Iris-virginica'],
		[6.3,2.9,5.6,1.8,'Iris-virginica'],
		[6.5,3.0,5.8,2.2,'Iris-virginica'],
		[7.6,3.0,6.6,2.1,'Iris-virginica'],
		[4.9,2.5,4.5,1.7,'Iris-virginica'],
		[7.3,2.9,6.3,1.8,'Iris-virginica'],
		[6.7,2.5,5.8,1.8,'Iris-virginica'],
		[7.2,3.6,6.1,2.5,'Iris-virginica'],
		[6.5,3.2,5.1,2.0,'Iris-virginica'],
		[6.4,2.7,5.3,1.9,'Iris-virginica'],
		[6.8,3.0,5.5,2.1,'Iris-virginica'],
		[5.7,2.5,5.0,2.0,'Iris-virginica'],
		[5.8,2.8,5.1,2.4,'Iris-virginica'],
		[6.4,3.2,5.3,2.3,'Iris-virginica'],
		[6.5,3.0,5.5,1.8,'Iris-virginica'],
		[7.7,3.8,6.7,2.2,'Iris-virginica'],
		[7.7,2.6,6.9,2.3,'Iris-virginica'],
		[6.0,2.2,5.0,1.5,'Iris-virginica'],
		[6.9,3.2,5.7,2.3,'Iris-virginica'],
		[5.6,2.8,4.9,2.0,'Iris-virginica'],
		[7.7,2.8,6.7,2.0,'Iris-virginica'],
		[6.3,2.7,4.9,1.8,'Iris-virginica'],
		[6.7,3.3,5.7,2.1,'Iris-virginica'],
		[7.2,3.2,6.0,1.8,'Iris-virginica'],
		[6.2,2.8,4.8,1.8,'Iris-virginica'],
		[6.1,3.0,4.9,1.8,'Iris-virginica'],
		[6.4,2.8,5.6,2.1,'Iris-virginica'],
		[7.2,3.0,5.8,1.6,'Iris-virginica'],
		[7.4,2.8,6.1,1.9,'Iris-virginica'],
		[7.9,3.8,6.4,2.0,'Iris-virginica'],
		[6.4,2.8,5.6,2.2,'Iris-virginica'],
		[6.3,2.8,5.1,1.5,'Iris-virginica'],
		[6.1,2.6,5.6,1.4,'Iris-virginica'],
		[7.7,3.0,6.1,2.3,'Iris-virginica'],
		[6.3,3.4,5.6,2.4,'Iris-virginica'],
		[6.4,3.1,5.5,1.8,'Iris-virginica'],
		[6.0,3.0,4.8,1.8,'Iris-virginica'],
		[6.9,3.1,5.4,2.1,'Iris-virginica'],
		[6.7,3.1,5.6,2.4,'Iris-virginica'],
		[6.9,3.1,5.1,2.3,'Iris-virginica'],
		[5.8,2.7,5.1,1.9,'Iris-virginica'],
		[6.8,3.2,5.9,2.3,'Iris-virginica'],
		[6.7,3.3,5.7,2.5,'Iris-virginica'],
		[6.7,3.0,5.2,2.3,'Iris-virginica'],
		[6.3,2.5,5.0,1.9,'Iris-virginica'],
		[6.5,3.0,5.2,2.0,'Iris-virginica'],
		[6.2,3.4,5.4,2.3,'Iris-virginica'],
		[5.9,3.0,5.1,1.8,'Iris-virginica']]

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
def giniimpurity(rows):
  total=len(rows)
  counts=uniquecounts(rows)
  imp=0
  for k1 in counts:
    p1=float(counts[k1])/total
    for k2 in counts:
      if k1==k2: continue
      p2=float(counts[k2])/total
      imp+=p1*p2
  return imp

# Entropy is the sum of p(x)log(p(x)) across all 
# the different possible results
def entropy(rows):
   from math import log
   log2=lambda x:log(x)/log(2)  
   results=uniquecounts(rows)
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

def buildtree(rows,scoref=entropy):
  if len(rows)==0: return decisionnode()
  current_score=scoref(rows)

  # Set up some variables to track the best criteria
  best_gain=0.0
  best_criteria=None
  best_sets=None
  
  column_count=len(rows[0])-1
  for col in range(0,column_count):
    # Generate the list of different values in
    # this column
    column_values={}
    for row in rows:
       column_values[row[col]]=1
    # Now try dividing the rows up for each value
    # in this column
    for value in column_values.keys():
      (set1,set2)=divideset(rows,col,value)
      
      # Information gain
      p=float(len(set1))/len(rows)
      gain=current_score-p*scoref(set1)-(1-p)*scoref(set2)
      if gain>best_gain and len(set1)>0 and len(set2)>0:
        best_gain=gain
        best_criteria=(col,value)
        best_sets=(set1,set2)
  # Create the sub branches   
  if best_gain>0:
    trueBranch=buildtree(best_sets[0])
    falseBranch=buildtree(best_sets[1])
    return decisionnode(col=best_criteria[0],value=best_criteria[1],
                        tb=trueBranch,fb=falseBranch)
  else:
    return decisionnode(results=uniquecounts(rows))
