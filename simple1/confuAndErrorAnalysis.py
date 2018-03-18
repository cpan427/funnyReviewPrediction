import numpy as np 
import pandas as pd 

embedArray = range(50)
revRead = pd.read_csv('reviews.csv', header=None, usecols=embedArray)
predRead = pd.read_csv('predictions.csv', header=None, usecols=[0])
actRead = pd.read_csv('truelabels.csv', header=None, usecols=[0])

getEmbeddings = open("data/small/words.txt", 'r')
embed = getEmbeddings.readlines()

rev = revRead.as_matrix()
pred = predRead.as_matrix()
act = actRead.as_matrix()

results = open("results.txt", "w")

ttl = len(rev)
truePos = 0
trueNeg = 0
falsePos = 0
falseNeg = 0

def getSentences(rev):
	res = ""
	#print ("Hi!")
	for n in rev:
		curr = embed[n]
		curr = curr.rstrip()
		if curr != "<pad>":
			res += curr + " "
	return res

print(len(rev))
print(len(pred))
print(len(act))


for i in range(len(pred)):
	currPred = pred[i][0]
	currAct = act[i][0]
	currRev = getSentences(rev[i])
	#print (str(currRev) + "\t" + str(currPred) + "\t" + str(currAct) + "\n")
	results.write(str(currRev) + "\t" + str(currPred) + "\t" + str(currAct) + "\n")
	if (str(currPred) == '1'):
		if (str(currAct) == '1'):
			truePos += 1
		else:
			falsePos += 1
	else:
		if (str(currAct) == '0'):
			trueNeg += 1
		else:
			falseNeg += 1

print("Confusion Matrix")
print("True Positives (pred: 1, act: 1)", truePos)
print("True Negatives (pred: 0, act: 0)", trueNeg)
print("False Positives (pred: 1, act: 0)", falsePos)
print("False Negatives (pred: 0, act: 1)", falseNeg)
