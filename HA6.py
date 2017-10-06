import nltk
from nltk.corpus import nps_chat as chat
import re
wordlist = [w for w in chat.words()]

#I'll choose the chat corpus.
#This corpus contains texts from chats.
#SMS is most likely used to submit chat messages.
#Our suggestions are most likely to be correct if they are based on a corpus
#from the same application domain.

def get_T9_word(digits):

    #create mapping
    digitLetterMap = {}
    digitLetterMap['0'] = ' '
    digitLetterMap['1'] = '["@-_]'
    digitLetterMap['2'] = '[AaBbCc]'
    digitLetterMap['3'] = '[DdEeFf]'
    digitLetterMap['4'] = '[GgHhIi]'
    digitLetterMap['5'] = '[JjKkLl]'
    digitLetterMap['6'] = '[MmNnOo]'
    digitLetterMap['7'] = '[PpQqRrSs]'
    digitLetterMap['8'] = '[TtUuVv]'
    digitLetterMap['9'] = '[WwXxYyZz]'

    #create regular expression
    expression = ''
    for digit in digits:
        expression = expression + digitLetterMap[digit] #expression will look like '[WwXxYyZz][MmNnOo][DdEeFf]'


    matches = nltk.FreqDist([w for w in wordlist if re.search('^'+expression+'$', w)]) #create the query expression and find matches in chat corpus.
    if(matches):
        return (matches.most_common(1)[0][0]) #return most common match

    return '' #return empty string if no matches have been found

#Test the function with following input
sentence = ['43556','73837','4','26','3463']
#Desired Output is 'hello peter i am fine'
for sequence in sentence:
    print(get_T9_word(sequence))

#actual output is:
    #hello
    #
    #i
    #am
    #find

#No matches have been found for the sequence '73837' which should have resulted im 'peter'. the name peter is not mentioned in the chat-corpus neither is any other word that would fit that sequence.
#For the sequence '3463' the actual output was 'found' instead of the expected 'fine'. the word 'found' was more common in the chat corpus, than fine.
