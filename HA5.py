import nltk
from nltk.corpus import wordnet as wn
import scipy

pairsString = 'car-automobile, gem-jewel, journey-voyage, boy-lad, coast-shore, asylum-madhouse, magician-wizard, midday-noon, furnace-stove, food-fruit, bird-cock, bird-crane, tool-implement, brother-monk, lad-brother, crane-implement, journey-car, monk-oracle, cemetery-woodland, food-rooster, coast-hill, forest-graveyard, shore-woodland, monk-slave, coast-forest, lad-wizard, chord-smile, glass-magician, rooster-voyage, noon-string'

#convert the string pairs to touples
pairsList = pairsString.split(', ')
touplesList = [(pair.split('-')[0], pair.split('-')[1])
               for pair in pairsList]

#calculate a similarity score and store it in a dictionary
scores = {}
for touple in touplesList:
    synset1 = wn.synsets(touple[0])
    synset2 = wn.synsets(touple[1])

    #calculate score for each touple by computing the path-similarity of each possible synset combination
    score = max(max(synset1[i].path_similarity(synset2[k])
                    for k in list(range(len(synset2))) if (synset1[i].path_similarity(synset2[k]) is not None))
                for i in list(range(len(synset1))))

    scores[touple] = score

#order data by descending similarity score
scores2 = list((1-value, key) for (key, value) in scores.items()) #This is done to enable correct sorting in the next step.
myRank = [touple for (score, touple) in sorted(scores2)] #sort the data by descending similarity score

print('Word-Pairs ranked by similarity:\n', myRank,'\n\n')#print out ranked data

#calculate similarity between given and own Rank
sim = scipy.stats.spearmanr([myRank.index(touple)
                                    for touple in touplesList],
                                   [touplesList.index(touple)
                                    for touple in touplesList])

print('Spearman Correlation between:')
print([touplesList.index(touple) for touple in touplesList])
print('and')
print([myRank.index(touple) for touple in touplesList])
print('results in:\n')
print(sim)
