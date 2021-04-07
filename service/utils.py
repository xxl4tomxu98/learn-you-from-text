import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.utils import shuffle
import re
from bs4 import BeautifulSoup
import pickle
import os
import glob
import numpy as np


def paragraph_to_words(paragraph):
    # preprocessing the user input text into tokens
    nltk.download("stopwords", quiet=True)
    stemmer = PorterStemmer()  
    text = BeautifulSoup(paragraph, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Convert to lower case
    words = text.split() # Split string into words
    words = [w for w in words if w not in stopwords.words("english")] # Remove stopwords
    words = [stemmer.stem(w) for w in words] # stem    
    return words


def convert_and_pad(word_dict, sentence, pad=500):
    # 500 pad is arbitary since normal text sentences are not long
    NOWORD = 0 # We will use 0 to represent the 'no word' category
    INFREQ = 1 # and we use 1 to represent infrequent words not appearing in word_dict    
    working_sentence = [NOWORD] * pad    
    for word_index, word in enumerate(sentence[:pad]):
        if word in word_dict:
            working_sentence[word_index] = word_dict[word]
        else:
            working_sentence[word_index] = INFREQ            
    return working_sentence, min(len(sentence), pad)


def convert_and_pad_data(word_dict, data, pad=500):
    # paragraph padding calling sentence padding 
    result = []
    lengths = []    
    for sentence in data:
        converted, leng = convert_and_pad(word_dict, sentence, pad)
        result.append(converted)
        lengths.append(leng)        
    return np.array(result), np.array(lengths)



def preprocess_data(data_train, data_test, labels_train, labels_test,
                    cache_dir=os.path.join("./static/cache", "sentiment_analysis"), 
                    cache_file="preprocessed_data.pkl"):
    """Convert each review to words; read from cache if available."""
    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay    
    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Preprocess training and test data to obtain words for each text entered
        #words_train = list(map(paragraph_to_words, data_train))
        #words_test = list(map(paragraph_to_words, data_test))
        words_train = [paragraph_to_words(text) for text in data_train]
        words_test = [paragraph_to_words(text) for text in data_test]        
        # Write to cache file for future runs
        if cache_file is not None:
            cache_data = dict(words_train=words_train, words_test=words_test,
                              labels_train=labels_train, labels_test=labels_test)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        words_train, words_test, labels_train, labels_test = (cache_data['words_train'],
                cache_data['words_test'], cache_data['labels_train'], cache_data['labels_test'])    
    return words_train, words_test, labels_train, labels_test


def prepare_imdb_data(data, labels):
    """Prepare training and test sets from IMDb movie reviews."""    
    #Combine positive and negative reviews and labels
    data_train = data['train']['pos'] + data['train']['neg']
    data_test = data['test']['pos'] + data['test']['neg']
    labels_train = labels['train']['pos'] + labels['train']['neg']
    labels_test = labels['test']['pos'] + labels['test']['neg']    
    #Shuffle reviews and corresponding labels within training and test sets
    data_train, labels_train = shuffle(data_train, labels_train)
    data_test, labels_test = shuffle(data_test, labels_test)    
    # Return a unified training data, test data, training labels, test labets
    return data_train, data_test, labels_train, labels_test


def read_imdb_data(data_dir='../data/aclImdb'):
    data = {}
    labels = {}    
    for data_type in ['train', 'test']:
        data[data_type] = {}
        labels[data_type] = {}        
        for sentiment in ['pos', 'neg']:
            data[data_type][sentiment] = []
            labels[data_type][sentiment] = []            
            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            files = glob.glob(path)            
            for f in files:
                with open(f) as review:
                    data[data_type][sentiment].append(review.read())
                    # Here we represent a positive review by '1' and a negative review by '0'
                    labels[data_type][sentiment].append(1 if sentiment == 'pos' else 0)                    
            assert len(data[data_type][sentiment]) == len(labels[data_type][sentiment]), \
                    "{}/{} data size does not match labels size".format(data_type, sentiment)                
    return data, labels


def build_dict(data, vocab_size = 5000):
    """Construct and return a dictionary mapping each of the most frequently 
    appearing words to a unique integer.    
    Determine how often each word appears in `data`. Note that `data` 
    is a list of sentences and that a sentence is a list of words. """
    
    word_count = {} # A dict storing the words that appear in the reviews along with how often they occur
    for phrase in data:
        for word in phrase:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1   
    # Sort the words found in `data` so that sorted_words[0] is the most frequently 
    # appearing word and sorted_words[-1] is the least frequently appearing word.    
    sorted_words = None
    sorted_words = sorted(word_count, key=word_count.get, reverse=True)    
    word_dict = {} # This is what we are building, a dictionary that translates words into integers
    for idx, word in enumerate(sorted_words[:vocab_size - 2]): # The -2 is so that we save room for the 'no word'
        word_dict[word] = idx + 2                              # 'infrequent' labels        
    return word_dict
