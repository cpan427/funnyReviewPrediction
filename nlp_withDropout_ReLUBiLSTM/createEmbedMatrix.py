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
#Note that this would have to be run on python3

import numpy as np
import pandas as pd

print ("Opening Glove pretrained vectors")
gloveEmbed = open("data/glove.6B.50d.txt", 'r', encoding='utf8')
embedDict = {}
embed = gloveEmbed.readlines()

print ("Creating dictionary")
for r in embed:
	r = r.split(" ")
	r[len(r)-1] = r[len(r)-1][:-1]
	val = r[0]
	r.remove(val)
	r = [float(x) for x in r]
	embedDict[val] = r

vocab = open("data/small/words.txt", 'r')
matrix = []

print ("Done with glove")
for w in vocab.readlines():
	curr = w
	if w[-1] == "\n":
		curr = w[:-1]
	currArr = None
	if curr in embedDict:
		currArr = embedDict[curr]
	else:
		currArr = [0.0 for i in range(50)]
	matrix.append(currArr)
print ("Writing the file")

res = np.asarray(matrix)
np.savetxt("data/small/embeddings.csv", res, delimiter=",")
