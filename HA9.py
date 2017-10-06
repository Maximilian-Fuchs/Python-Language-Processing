import nltk
from nltk.corpus import movie_reviews
from nltk.tag.stanford import StanfordTagger
from nltk.tag import pos_tag
import random

# prepare review data as list of touples
# category is positive / negative

review_data = [(movie_reviews.words(fileid), category)
               for category in movie_reviews.categories()
               for fileid in movie_reviews.fileids(category)]
#review_data looks like [([token, token, ...], pos/neg), ...]
#random.shuffle(review_data)   #commented for performance

threshold = 1000

fd_all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
top_words = [word for (word, count) in fd_all_words.most_common(threshold)]
# top_words looks like  [(',', 77717), ('the', 76529), ('.', 65876), ('a', 38106), ('and', 35576), ('of', 34123), ('to', 31937), ("'", 30585), ('is', 25195), ('in', 21822)]

#review_data_fdist = [
#    (nltk.FreqDist(token.lower() for token in words if token in top_words),
#     category)
#    for words, category in review_data]

postagged_review_data = [pos_tag(review, tagset='universal') for (review, category) in review_data]
fd_adjectives = nltk.FreqDist(word.lower() for tagged_review  in postagged_review_data
                               for (word,tag) in tagged_review if tag == 'ADJ')
top_adjectives = set(fd_adjectives.most_common(100))





# reimplementation of the books approach
def feature_extractor1(review):
    review_words = set(review)
    features = {}
    for word in list(fd_all_words)[:2000]:
        features['contains({})'.format(word)] = (word in review_words)
    return features

featuresets1 = [(feature_extractor1(d), c) for (d,c) in review_data]
train_set1, test_set1 = featuresets1[100:], featuresets1[:100]
classifier1 = nltk.NaiveBayesClassifier.train(train_set1)


#we improve the first approach by not considering the most frequent words. We assume that more frequent words do not characterize the data very well.
# results show only slight improvement. Maybe because Bayesian Classifier is already aware of words which hold few information about the class.
def feature_extractor3(review):
    review_words = set(review)
    features = {}
    for word in list(fd_all_words)[:2000]:
        if word not in top_words:
            features['contains({})'.format(word)] = (word in review_words)
    return features

featuresets3 = [(feature_extractor3(d), c) for (d,c) in review_data]
train_set3, test_set3 = featuresets3[100:], featuresets3[:100]
classifier3 = nltk.NaiveBayesClassifier.train(train_set3)





# this featureset only considers words tagged as Adjective. We assume that most information about the class resides in the adjectives.
# It showed way better Accuracy than the plain word-feature approach from the book
def feature_extractor2(review):
    review_words = set(review)
    features = {}
    for tagged_review in postagged_review_data:
        for (word, tag) in tagged_review:
            if(tag == 'ADJ'):
                features['contains({})'.format(word)] = (word in review_words)
    return features

featuresets2 = [(feature_extractor2(d), c) for (d,c) in review_data]
train_set2, test_set2 = featuresets2[100:], featuresets2[:100]
classifier2 = nltk.NaiveBayesClassifier.train(train_set2)







# We combine the 2 earlier approaches and exclude the most frequent adjectives.
def feature_extractor4(review):
    review_words = set(review)
    features = {}
    for tagged_review in postagged_review_data:
        for (word, tag) in tagged_review:
            if(tag == 'ADJ'):
                if word not in top_adjectives:
                    features['contains({})'.format(word)] = (word in review_words)
    return features

featuresets4 = [(feature_extractor4(d), c) for (d,c) in review_data]
train_set4, test_set4 = featuresets4[100:], featuresets4[:100]
classifier4 = nltk.NaiveBayesClassifier.train(train_set4)



#output the results
print("\n\n",nltk.classify.accuracy(classifier1, test_set1))
classifier1.show_most_informative_features(5)

print("\n\n",nltk.classify.accuracy(classifier2, test_set2))
classifier2.show_most_informative_features(5)

print("\n\n",nltk.classify.accuracy(classifier3, test_set3))
classifier3.show_most_informative_features(5)

print("\n\n",nltk.classify.accuracy(classifier4, test_set4))
classifier4.show_most_informative_features(5)
