'''
Created on Feb 11, 2017

@author: mozhu
'''

import numpy as np
import csv as csv
import itertools
import nltk as nltk
from vilinarnn.RNNNumpy import RNNNumpy

#nltk.download("book")

vocabulary_size = 4000
unknow_token = "UNKNOW_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

print("read data from file ")

sentences= []
with open('C:\\Users\\mozhu\\Desktop\\training.csv',encoding='utf8') as f:
    reader = csv.reader(f, skipinitialspace = True)
    next(reader)
    # split full book into sentences
    '''
    for x in reader:
        print(x)
        print(x[0].lower())
    '''
    for x in reader:
        sentence = nltk.sent_tokenize(x[0].lower())
        sentences.extend(["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentence])
    #sentence = itertools.chain(*[nltk.sent_tokenize(x[0].lower() for x in reader)])
    # Append SENTENCE_START & SENTENCE_END
    
print("Parsed %d sentences." % (len(sentences)))

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
'''
for sent in sentences:
    print(sent)
    print("/n")
    tokenized_sentences.append(nltk.word_tokenize(sent[0]))
'''

#Count the work frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found %d unique words tokens." % len(word_freq.items()))

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size - 1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknow_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print("Using vocabulary size %d")
print("The least frequent word in our vocablary is '%s' and appeard %d times." % (vocab[-1][0], vocab[-1][1])) 

# replace all words not in our vocabulary with unknow token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknow_token for w in sent]
    
print("\nExample sentence: '%s'" % sentences[0])
print("\nExample sentence after pre-process: '%s'" % tokenized_sentences[0])

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
Y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])


# test run for check
np.random.seed(10)

model = RNNNumpy(vocabulary_size)
model.train_with_sgd(model, X_train[:100], Y_train[:100], nepoch = 10, evaluate_loss_after = 1)




