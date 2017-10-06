# By replacing special characters that are considered by the spam classifier
#with similiar UTF-8 Characters I was abled lo lower the Precision, Recall and F-Score by following amounts:
#Precision was lowered by 0,231 to 0,632
#Recall was lowered by 0,444 to 0,529
#F-Measure was lowered by 0,340 to 0,576

#This file will output an altered email corpus 'newcorpus' to the directory it's run in.
#To test this corpus copy the newcorpus to your nltk corpora folder and execute the run_test.py script.

import nltk, random, logging, math
from corpus_mailsModule import mails, alteredmails
from t10EvaluationModule import compute_PRF
from t10SpamClassifierModule import get_feature_set, evaluate
logging.basicConfig(level=logging.DEBUG)
import os
import random

def preprocess_mails(email_data):
    altered_email_data = []

    for id_and_category in email_data:
        fileid = id_and_category[0]
        category = id_and_category[1]
        mailtext = mails.raw(fileid)
        if category == 'spam':
            altered_mail_text = ""

            for charpos in range(len(mailtext)):
                if mailtext[charpos] == "!":
                    altered_mail_text += '‼'
                    continue
                if mailtext[charpos] == "$":
                    altered_mail_text += '＄'
                    continue
                if mailtext[charpos] == '*':
                    altered_mail_text += '﹡'
                    continue
                if mailtext[charpos] == '-':
                    altered_mail_text += '－'
                    continue
                if mailtext[charpos] == 'h' and mailtext[charpos+1] == 't' and mailtext[charpos+2] == 't' and mailtext[charpos+3] == 'p':
                    altered_mail_text += '_h'
                    continue
                altered_mail_text += mailtext[charpos]



            altered_email_data.append((altered_mail_text, fileid, category))
        else:
            altered_email_data.append((mailtext, fileid, category))


    return altered_email_data

def create_altered_corpus(altered_corpus_data):

    set_of_categories = set([category for (text, fileid, category) in altered_corpus_data])
    corpus =  [(text, fileid) for (text, fileid, category) in altered_corpus_data]

    corpusdir = 'newcorpus/'
    if not os.path.isdir(corpusdir):
        os.mkdir(corpusdir)

    # Make new dir for the corpus.
    for current_category in set_of_categories:
        corpusdir = 'newcorpus/' + current_category + '/'
        if not os.path.isdir(corpusdir):
            os.mkdir(corpusdir)
    corpusdir = 'newcorpus/'
    # Output the files into the directory.
    for (text, fileid, category) in altered_corpus_data:
        with open(corpusdir+str(fileid)+'_altered.txt', encoding='utf-8',mode = 'w') as fout:
            fout.write(text)

email_data = [(fileid, category)
    for category in mails.categories()
    for fileid in mails.fileids(category)]
create_altered_corpus(preprocess_mails(email_data))
