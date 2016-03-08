import nltk
import numpy as np


words = corpus.split()
cfreq_sam = nltk.ConditionalFreqDist(nltk.bigrams(words))
cprob_sam = nltk.ConditionalProbDist(cfreq_sam, nltk.MLEProbDist)

class WordPredict(object):

    def __init__(self):

        pass


    def fit(self):
        # split the word into bigrams


        # put into matrix

        # smoothing




    def predict(self, preced_words):
        # split the preced_words into bigrams

        # get probability with all the bigrams

        words = corpus.split()
        cfreq_sam = nltk.ConditionalFreqDist(nltk.bigrams(words))
        cprob_sam = nltk.ConditionalProbDist(cfreq_sam, nltk.MLEProbDist
        # product of all words probability(preced_word)


        # get vector for the last word in preced
