import pandas as pd
import numpy as np
import re
import random as rand

pattern = re.compile('\n')

print ("Starting...")

rev = pd.read_csv('yelp_review.csv')

text = rev['text'].as_matrix()
fun = rev['funny'].as_matrix()

wordsTest = open("text_test.txt", "w")
revsTest = open("answers_test.txt", "w")
succTestCount = 0

wordsDev = open("text_dev.txt", "w")
revsDev = open("answers_dev.txt", "w")
succDevCount = 0

wordsTrain = open("text_train.txt", "w")
revsTrain = open("answers_train.txt", "w")
succCount = 0

excluded = 0

print ("Looking at sentences")

sentLens = []

def chooseBatch(wordsTest, revsTest, wordsDev, revsDev, wordsTrain, revsTrain):
	num = rand.random()
	if (num <= 250000/5200000):
		return wordsTest, revsTest, 0
	elif (num <= 500000/5200000):
		return wordsDev, revsDev, 1
	return wordsTrain, revsTrain, 2


for i in range(len(text)):
	if (i % 100000 == 0):
		print ("Finished", i, "reviews")
	try:
		wrd = re.sub(pattern, '', text[i])
		wrd.encode('ascii')

		sent = wrd.split(" ")
		if (len(sent) <= 500):
			sentLens.append(len(sent))

			words, revs, whichSet = chooseBatch(wordsTest, revsTest, wordsDev, revsDev, wordsTrain, revsTrain)

			words.write(wrd + "\n")
			ans = 0
			if (fun[i] > 0):
				ans = 1
			revs.write(str(ans) + "\n")
			
			if (whichSet == 0):
				succTestCount += 1
			elif (whichSet == 1):
				succDevCount += 1
			else:
				succCount += 1
		else:
			excluded += 1
	except UnicodeEncodeError:
		excluded += 1

print ("Cool")
print ("Number of rows for train:", succCount)
print ("Number of rows for dev:", succDevCount)
print ("Number of rows for test:", succTestCount)
print ("Number of excluded (long and non-ASCII and non English) reviews:", excluded)
print ("Total Number of reviews included", (len(text) - excluded))
print (np.percentile(sentLens,[0, 25, 50, 75, 80, 90, 95, 100]))