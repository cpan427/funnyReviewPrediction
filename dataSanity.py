import pandas as pd
import numpy as np
import re
import random as rand

wordsTest = open("text_test.txt", "r")
revsTest = open("answers_test.txt", "r")

wordsDev = open("text_dev.txt", "r")
revsDev = open("answers_dev.txt", "r")

wordsTrain = open("text_train.txt", "r")
revsTrain = open("answers_train.txt", "r")

print(len(wordsTest.readlines()), len(revsTest.readlines()))
assert(len(wordsTest.readlines()) == len(revsTest.readlines()))

print(len(wordsDev.readlines()), len(revsDev.readlines()))
assert(len(wordsDev.readlines()) == len(revsDev.readlines()))

print(len(wordsTrain.readlines()), len(revsTrain.readlines()))
assert(len(wordsTrain.readlines()) == len(revsTrain.readlines()))