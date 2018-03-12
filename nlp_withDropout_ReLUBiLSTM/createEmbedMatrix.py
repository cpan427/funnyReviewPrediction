"""
createEmbeddingMatrix:

Pseudocode:
- open the matrix //Note that we should just make it an embedding
   - make this into a dictionary
- go through the vocab in order (read the vocab file)
- if the word is not in the matrix
   - create a random embedding
- else:
   - append the word embedding


//Read the thing into a numpy array by doing genfromtxt(__, delimiter=',')
"""

import numpy as np
import pandas as pd

gloveEmbed = open("glove.6B.50d.txt", 'r', encoding='utf8')
embedDict = {}
embed = gloveEmbed.readlines()
for r in embed:
	r = r.split(" ")
	r[len(r)-1] = r[len(r)-1][:-1]
	val = r[0]
	r.remove(val)
	r = [float(x) for x in r]
	embedDict[val] = r

vocab = open("words.txt", 'r')
matrix = []

for w in vocab.readlines():
	curr = w[:-1]
	currArr = None
	if curr in embedDict:
		currArr = embedDict[curr]
	else:
		#print ("We tried....")
		currArr = [0.0 for i in range(50)]
	matrix.append(currArr)
#print (matrix)
res = np.asarray(matrix)
np.savetxt("embeddingMatrix.csv", res, delimiter=",")