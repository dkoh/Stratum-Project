import treepredict, treepredictstratum, csv


def tofloat(data1,columns):
	for row in data1:
		for column in columns:
			row[column]= float(row[column])
	return data1

dummydata = list(csv.reader(open("dummydata.csv", "rb")))
# predictors=[]
# response=[]
# for row in dummydata:
# 	predictors.append(row[:2])
# 	response.append(row[2:4])
dummydata=tofloat(dummydata,range(len(dummydata[0])/2))




#Testing program
col=0
column_values={}
for row in dummydata:
	column_values[row[col]]=1
# Now try dividing the rows up for each value
# in this column
for value in column_values.keys():
	(set1,set2)=treepredictstratum.divideset(dummydata,col,value)
	p=float(len(set1))/len(dummydata)
	if value==4.5: print treepredictstratum.entropy(set1,2), treepredictstratum.entropy(set2,2)
	
'''

tree=treepredictstratum.buildtree(dummydata)
treepredictstratum.printtree(tree)


dummydata1=[]
for row in dummydata:
	result=row[:2]
	if row[2]=='0' and row[3]=='0': result.append('1')
	if row[2]=='0' and row[3]=='1': result.append('2')
	if row[2]=='1' and row[3]=='0': result.append('3')
	if row[2]=='1' and row[3]=='1': result.append('4')	
	dummydata1.append(result)

tree=treepredict.buildtree(dummydata1)
#treepredict.printtree(tree)
'''