import nltk
from nltk.corpus import nps_chat
from nltk.corpus import brown
import random
from nltk.tag.perceptron import PerceptronTagger




def preprocess(posts):
    for i in range(len(posts)):
        for j in range(len(posts[i])):
            if posts[i][j][0] == '':
                posts[i][j] = (' ', posts[i][j][1])
    return posts



tagged_posts = list(nps_chat.tagged_posts(tagset = 'universal'))
nr_posts = len(tagged_posts)
random.shuffle(tagged_posts)
train_posts = list(tagged_posts[:(nr_posts*8)//10]) #first 80% of corpus
developement_posts = list(tagged_posts[(nr_posts*8)//10:(nr_posts*9)//10]) # 80-90% of the corpus
test_posts = list(tagged_posts[((nr_posts*9)//10):])


# using the PerceptronTagger according to nltk documentation
perc = PerceptronTagger(load=False)
perc.train(preprocess(preprocess(train_posts + developement_posts))) #using the developement_posts for training in production
score = perc.evaluate(preprocess(test_posts)) #preprocessing is necessary to eliminate strings with length 0 which will crash the PerceptronTagger
print(score)
