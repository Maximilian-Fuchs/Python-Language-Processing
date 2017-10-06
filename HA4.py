import nltk
from nltk.corpus import udhr, genesis
from nltk.probability import FreqDist
from nltk.probability import ConditionalFreqDist
import math
import numpy


# Homework 4.1
#Implement a function
#build_language_models(languages,words)
#which takes a list of languages and a
#dictionary of words as arguments and returns a conditional frequency distribution where:
#     the languages are the conditions
#     the values are the lower case characters found in
#words[language]
#Call the function as follows:
def build_language_models(languages, words):
    cfd = nltk.ConditionalFreqDist(
            (language, ch) #FreqDist of Characters per Language
            for language in languages
            for ch in words[language]
            if ch.isalpha() and ch.islower()) # only lower chars

    return cfd


# Code according to Exercise sheet
languages = ['English', 'German_Deutsch', 'French_Francais']
language_base = dict((language, ''.join(udhr.words(language + '-Latin1'))) for language in languages) # changed the dictionary to contain string of chars instead of lists of words
language_model_cfd = build_language_models(languages, language_base)
for language in languages:
        for key in list(language_model_cfd[language].keys())[:10]:
            print(language, key, '->', language_model_cfd[language].freq(key))





def get_language_similarity_score(language_model_cfd, text):

    languages = language_model_cfd.conditions()

    textCharCount = FreqDist(char for char in text if char.islower()) #Build the FreqDist for lower chars in text

    #removing the most common value from the text-FreqDist has shown better results.
    for (commonletter, value) in textCharCount.most_common(1):
        del textCharCount[commonletter]

    scores = {}

    #for every language compute the correlation coefficient of the two frequencyDistributions
    #correlation coefficient can be interpreted as a measure of the 'similarity' of two functions.
    # computation corresponds to https://wikimedia.org/api/rest_v1/media/math/render/svg/5c0b628a0b05e6bce02398c0a26567504098d4f8 where x1, x2, ..., xn are the character frequencys of the given text and y1, y2, ... are the character frequencys of the language model and "x_" and "y_" correspond to https://wikimedia.org/api/rest_v1/media/math/render/svg/b85e0dc181a9d887e648ec3ac64188d14193619a
    for language in languages:

        a = [language_model_cfd[language].freq(key) for key in language_model_cfd[language].keys()] #make list from frequencies in language model.
        b = [textCharCount.freq(key) for key in language_model_cfd[language].keys()] #make list from frequencies in text. Use same keys as before, so the position of the frequencies correspond to each other.

        scores[language] = numpy.corrcoef(a,b)[0][1] #compute correlation coeffient of text and language model for each language-model. Save to a dictionary.

    return scores



def guess_language(language_model_cfd, text):
    scores = get_language_similarity_score(language_model_cfd, text) #compute score
    scores2 = dict((value, key) for (key, value) in scores.items())  #switch value and key so we can sort by scores in the next step
    return scores2[max(scores2.keys())] # return language-string with highest score


print('Based on Character-Frequency')
languages = ['English', 'German_Deutsch', 'French_Francais']
language_base = dict((language, ''.join(udhr.words(language + '-Latin1'))) for language in languages)
language_model_cfd = build_language_models(languages, language_base)

text1 = "Peter had been to the office before they arrived."
text2 = "Si tu finis tes devoirs, je te donnerai des bonbons."
text3 = "Das ist ein schon recht langes deutsches Beispiel."

# guess the language by comparing the frequency distributions
print('guess for english text is', guess_language(language_model_cfd, text1))
print('guess for french text is', guess_language(language_model_cfd, text2))
print('guess for german text is', guess_language(language_model_cfd, text3))








#
#
#
#The previous language guesser was based on the frequency of characters.  Implement alternative language guesser
#based on the following lexical units:

#4.1  Homework
#(a)  tokens
#(b)  character bigrams
#(c)  token bigrams
#
#



# The language guesser function was altered to call altered versions of get_language_similarity_score and build_language_models
# The alternate versions have the digit 2 appended to the function name : get_language_similarity_score2 and build_language_models2
# The functions were modified to not take lower characters only anymore
# Modifications consisted of: adjust calls to renamed method; remove checks for .islower(); check if FreqDists share any key befor calculating Correlation Coefficient to prevent division by zero


def guess_language2(language_model_cfd, text):
    scores = get_language_similarity_score2(language_model_cfd, text)
    scores2 = dict((value, key) for (key, value) in scores.items())
    return scores2[max(scores2.keys())]


def build_language_models2(languages, words):
    cfd = nltk.ConditionalFreqDist(
            (language, ch) #FreqDist of Characters per Language
            for language in languages
            for ch in words[language])


    return cfd

def get_language_similarity_score2(language_model_cfd, text):

    languages = language_model_cfd.conditions()

    textCharCount = FreqDist(char for char in text) #Build the FreqDist for lower chars in text

    #removing the most common value from the text-FreqDist has shown better results.
    for (commonletter, value) in textCharCount.most_common(1):
        del textCharCount[commonletter]

    scores = {}

    #for every language compute the correlation coefficient of the two frequencyDistributions
    #correlation coefficient can be interpreted as a measure of the 'similarity' of two functions.
    for language in languages:

        if(any(key in textCharCount.keys() for key in language_model_cfd[language].keys())): #check if FreqDists share any key to prevent division by zero
            a = [language_model_cfd[language].freq(key) for key in language_model_cfd[language].keys()]
            b = [textCharCount.freq(key) for key in language_model_cfd[language].keys()]
            scores[language] = numpy.corrcoef(a,b)[0][1] #compute correlation coeffient of text and language model for each language-model. Save to a dictionary.
        else:
            scores[language] = 0


    return scores


#(a)  tokens
print('Based on Token-Frequency')
languages = ['English', 'German_Deutsch', 'French_Francais']
language_base = dict((language, udhr.words(language + '-Latin1')) for language in languages)
language_model_cfd = build_language_models(languages, language_base)

text1 = "Peter had been to the office before they arrived."
text2 = "Si tu finis tes devoirs, je te donnerai des bonbons."
text3 = "Das ist ein schon recht langes deutsches Beispiel."

# guess the language by comparing the frequency distributions
print('guess for english text is', guess_language2(language_model_cfd, text1.split(" ")))
print('guess for french text is', guess_language2(language_model_cfd, text2.split(" ")))
print('guess for german text is', guess_language2(language_model_cfd, text3.split(" ")))






#(b)  character bigrams
print('Based on characterbigrams-Frequency')
languages = ['English', 'German_Deutsch', 'French_Francais']
language_base = dict((language, nltk.bigrams(''.join(udhr.words(language + '-Latin1')))) for language in languages)
language_model_cfd = build_language_models2(languages, language_base)

text1 = "Peter had been to the office before they arrived."
text2 = "Si tu finis tes devoirs, je te donnerai des bonbons."
text3 = "Das ist ein schon recht langes deutsches Beispiel."

# guess the language by comparing the frequency distributions
print('guess for english text is', guess_language2(language_model_cfd, nltk.bigrams(''.join(text1))))
print('guess for french text is', guess_language2(language_model_cfd, nltk.bigrams(''.join(text2))))
print('guess for german text is', guess_language2(language_model_cfd, nltk.bigrams(''.join(text3))))





#(c)  token bigrams
print('Based on Token bigrams-Frequency')
languages = ['English', 'German_Deutsch', 'French_Francais']
language_base = dict((language, nltk.bigrams(' '.join(udhr.words(language + '-Latin1')).split(' '))) for language in languages)
language_model_cfd = build_language_models2(languages, language_base)

text1 = "Peter had been to the office before they arrived."
text2 = "Si tu finis tes devoirs, je te donnerai des bonbons."
text3 = "Das ist ein schon recht langes deutsches Beispiel."

# guess the language by comparing the frequency distributions
print('guess for english text is', guess_language2(language_model_cfd, nltk.bigrams(' '.join(text1.split(' ')))))
print('guess for french text is', guess_language2(language_model_cfd, nltk.bigrams(' '.join(text2).split(' '))))
print('guess for german text is', guess_language2(language_model_cfd, nltk.bigrams(' '.join(text3).split(' '))))



t41 = "Discuss, why English and German texts are difficult to distinguish with the given approach.\n\n Probably because they share a similar character frequency distribution and they also have a more similar alphabet as german and french or english and french.\n\n\n\n"
t42 = "Discuss, which approach should work best theoretically. Is this reflected in the results?\n\n Bigrams are more unique to a language than characters. But there are much more different Bigrams than characters. Therefore a classifier would need a very large model. So theoretically a tradeoff between characters and Bigrams should give us the best results when dealing with a limited model as in this case. The results reflect that."
print(t41,t42)
