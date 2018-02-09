"""
How to Develop a Word Embedding Model for Predicting Movie Review Sentiment
source: https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/
"""

from nltk.corpus import stopwords
import string
from os import listdir
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import Sequential


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# turn a doc into clean tokens
def clean_doc(doc, vocab=None):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    if vocab is None:
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if w not in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]
    else:
        # filter out tokens not in vocab
        tokens = [w for w in tokens if w in vocab]
        tokens = ' '.join(tokens)
    return tokens


# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
    # load doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # update counts
    vocab.update(tokens)


# load all docs in a directory
def process_docs(directory, vocab, is_train):
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # add doc to vocab
        add_doc_to_vocab(path, vocab)


# load all docs in a directory
def process_docs2(directory, vocab, is_train):
    documents = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load the doc
        doc = load_doc(path)
        # clean doc
        tokens = clean_doc(doc, vocab)
        # add to list
        documents.append(tokens)
    return documents


# save list to file
def save_list(lines, filename):
    # convert lines to a single blob of text
    data = '\n'.join(lines)
    # open file
    file = open(filename, 'w')
    # write text
    file.write(data)
    # close file
    file.close()


if __name__ == '__main__':
    # == == == == == == Part 1: pre process data == == == == == == #
    # define vocab
    vocab = Counter()
    # add all docs to vocab
    process_docs('./data/movie_review/txt_sentoken/neg', vocab, True)
    process_docs('./data/movie_review/txt_sentoken/pos', vocab, True)
    # print the size of the vocab
    print(len(vocab))
    # print the top words in the vocab
    print(vocab.most_common(50))
    # keep tokens with a min occurrence
    min_occurrence = 2
    tokens = [k for k,c in vocab.items() if c >= min_occurrence]
    print(len(tokens))
    # save tokens to a vocabulary file
    save_list(tokens, './data/movie_review/vocab.txt')

    # == == == == == == Part 2: prepare to train == == == == == == #
    # load the vocabulary
    vocab_filename = './data/movie_review/vocab.txt'
    vocab = load_doc(vocab_filename)
    vocab = vocab.split()
    vocab = set(vocab)

    # load all training reviews
    positive_docs = process_docs2('./data/movie_review/txt_sentoken/pos', vocab, True)
    negative_docs = process_docs2('./data/movie_review/txt_sentoken/neg', vocab, True)
    train_docs = negative_docs + positive_docs
    print(len(train_docs))
    print(train_docs[0])

    # create the tokenizer
    tokenizer = Tokenizer()
    # fit the tokenizer on the documents
    tokenizer.fit_on_texts(train_docs)

    # sequence encode
    encoded_docs = tokenizer.texts_to_sequences(train_docs)
    print(encoded_docs[0])
    max_length = max([len(s.split()) for s in train_docs])
    Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print(Xtrain[0, :])
    print(len(train_docs[0]), len(Xtrain[0, :]), len(encoded_docs[0]))
    # define training labels
    ytrain = np.array([0 for _ in range(900)] + [1 for _ in range(900)])
    print('-'*79)

    # load all test reviews
    positive_docs = process_docs2('./data/movie_review/txt_sentoken/pos', vocab, False)
    negative_docs = process_docs2('./data/movie_review/txt_sentoken/neg', vocab, False)
    test_docs = negative_docs + positive_docs
    # sequence encode
    encoded_docs = tokenizer.texts_to_sequences(test_docs)
    # pad sequences
    Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    # define test labels
    ytest = np.array([0 for _ in range(100)] + [1 for _ in range(100)])

    # == == == == == == Part 3: start training == == == == == == #
    # define vocabulary size (largest integer value)
    vocab_size = len(tokenizer.word_index) + 1

    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    print('-'*79)

    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(Xtrain, ytrain, epochs=10, verbose=2)
    # evaluate
    loss, acc = model.evaluate(Xtest, ytest, verbose=0)
    print('Test Accuracy: %f' % (acc * 100))
