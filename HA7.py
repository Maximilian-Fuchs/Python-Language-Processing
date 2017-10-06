import nltk
from nltk.corpus import brown

taggedWords = nltk.corpus.brown.tagged_words() #load list of (word, tag)-touples from brown-corpus
myList = [word for (word, tag) in taggedWords if tag == 'MD'] #create a list of words, tagged as 'MD'

#print a alphabetically sorted list of distinct words
print('\n\n10 of the words in Brown Corpus tagged as MD \n')
print(sorted(set(myList))[:10])

#make a dictionary that maps a word to the tags that it is associated with throughout the brown corpus
tagIndex = nltk.defaultdict(set)
for (word, tag) in taggedWords:
    tagIndex[word].add(tag)

#get dictionary keys where the list contains PLOURALOUN (NNS) and SINGULARVERB(VBZ)
results = [word for word in tagIndex.keys() if 'NNS' in tagIndex[word] and 'VBZ' in tagIndex[word]]
print('\n\n10 of the words that can be ploural nouns or third person verbs:\n')
print(results[:10])


#Identify three-word prepositional phrases of the form ADP + DET + NOUN (eg. "at the end")
taggedWords = nltk.corpus.brown.tagged_words(tagset='universal') #Switch to universal tag set
twpp = [] #three word prepositional phrases will be listed in twpp

#iterate through tagged words and look for a sequence of words tagged as ADP, DET, NOUN and add findings to 'twpp'
#this method assumes that every sentence and document in the brown corpus ends with a punctuation symbol.
for i in range(len(taggedWords)):
    if i > 1 and i < len(taggedWords)-1:
        if taggedWords[i-1][1] == 'ADP' and taggedWords[i][1] == 'DET' and taggedWords[i+1][1] == 'NOUN':
            twpp.append(taggedWords[i-1][0] + ' ' + taggedWords[i][0] + ' ' + taggedWords[i+1][0])

print('\n\n10 of the three word prepositional phrases:\n')
print(twpp[:10])


#What is the ratio of masculine to feminine pronouns?
m_pron_count = 0.0
f_pron_count = 0.0
#count occourences of masculine/feminine pronouns and calculate the ratio
for (word, tag) in taggedWords:
    if(tag == 'PRON'):
        if(word == 'he' or word == 'his' or word == 'him'):
            m_pron_count = m_pron_count + 1.0
        elif(word == 'she' or word == 'hers' or word == 'him'):
            f_pron_count = f_pron_count + 1.0
ratio = m_pron_count/f_pron_count

print('\n\nThe ratio of masculine to feminine pronouns: ')
print ratio


#Print a table with the integers 1..10 in one column, and the number of distinct words in the corpus having 1..10 distinct tags in the other column.
lenList = [len(tags) for tags in tagIndex.values()] #count number of treebank-tags for each word
fdist1 = nltk.FreqDist(lenList) #count words with same amount of tags

#print a 'table'
print('\n\nNumber of words having 1-10 Tags:\n')
for i in range(1, 11):
    print(i, '    ', fdist1[i])


#For the word(s) with the greatest number of distinct tags, print the sentences from the corpus containing the word, one for each possible tag.
tenTags = [(word, tags) for (word, tags) in tagIndex.items() if len(tags) == 10] #get a list of (word, tag)-touples of words with exactly 10 treebank-tags
sb = [] #example sentences will be put in this list

#go through brown tagged_senteces, check if sentence contains an example sentence for a (word, tag)-touple in 'tenTags'-list. if so add that sentence to 'sb'-list. Only one sentence per touple.
for sent in nltk.corpus.brown.tagged_sents():
    if len(tenTags[0][1]) == 0 and len(tenTags[1][1]) == 0:
        break
    for (word, tags) in tenTags:
        for touple in sent:
            if word == touple[0] and touple[1] in tags:
                sb.append(sent)
                tags.remove(touple[1])

print('\n\nExample Sentences for the Different Meanings of the most versatile words:\n')
for sent in sb:
    print(' '.join([word for (word, tag) in sent]) + '\n')
