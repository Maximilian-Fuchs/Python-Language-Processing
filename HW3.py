import nltk
from nltk.corpus import brown
from nltk.probability import FreqDist
import math

#Brown Corpus from nltk.corpus will be the Background Corpus, the text "Coping with Runaway Technology" will be the Foreground Corpus
fdist_bg = FreqDist(list(nltk.bigrams(brown.words())))
fdist_fg = FreqDist(list(nltk.bigrams(brown.words(fileids=['cg22']))))

def compute_LL(phrase,fdist_fg,fdist_bg):
# the variables are defined according to the Homework Handout
    a = fdist_fg[phrase]
    b = fdist_bg[phrase]
    c = fdist_fg.N()
    d = fdist_bg.N()
    n = c + d
    e1 = (c*(a+b))/n
    e2 = (d*(a+b))/n
    return 2*( a*math.log(a/e1, 2) + b*math.log(b/e2, 2) ) #the computation corresponds to the Log-Likelyhood formular given on the Homework Handout


scoreAndPhrase = [(compute_LL(phrase, fdist_bg, fdist_fg) ,phrase)   #create a list of touples that contain a phrase and it's Log-Likelyhood.
                  for phrase in set(list(nltk.bigrams(brown.words(fileids=['cg22']))))] #use bigrams from the text "Coping with Runaway Technology" as phrases.


print(list(reversed(sorted(scoreAndPhrase)))[:10]) #output the 10 most improbable phrases
