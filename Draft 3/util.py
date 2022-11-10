import pickle
from collections import OrderedDict
from os.path import exists
from typing import List

import nltk
import numpy as np
import torch
from nltk import word_tokenize

from string import punctuation

import re

from tqdm import tqdm

import gensim.downloader

nltk.download('brown')
punctuation = set(list(punctuation))
context_size = 10

data_file = "training_data.pkl"
word2index_file = "word2index.pkl"

words_to_search = { 'love', 'amazing', 'good', 'hate', 'sad', 'happy',
                        'bad', 'hurt', 'awesome', 'excited', 'nice', 'great', 'sick'}


class GloveTweetVectors:

    def __init__(self):
        self.gv = gensim.downloader.load('glove-twitter-25')
        self.zero_vector = np.zeros_like(self.gv.get_vector('is'))
        pass

    def get_word_to_vec_dict(self, words):

        return {
            word: self.gv.get_vector(word) if self.gv.has_index_for(word) else self.zero_vector
            for word in words
        }


def get_not_sarcasm_data():

    with open("not_sarcasm_data_3M.pkl", "rb") as f:
        sentences = pickle.load(f)

    return tweets_pre_processing([sent.content for sent in sentences])


def get_context_windows(sentences):

    #sentences = get_not_sarcasm_data()

    context_window_list = []
    for sent in tqdm(sentences):
        sent = [word.lower() for word in sent if (word.isalpha()) and (word not in punctuation)]
        for i, word in enumerate(sent[:-1]):
            context_window_list.append(tuple(sent[i:min(len(sent), i+context_size+1)]))

    return context_window_list


def get_data():

    if not (exists(data_file) and exists(word2index_file)):

        sents = get_not_sarcasm_data()

        words = [word for sent in sents for word in sent]
        words = [word.lower() for word in words]
        words = [word for word in words if word.isalpha()]
        words = [word for word in words if word not in punctuation]

        f_words = nltk.FreqDist(words)
        print(len(f_words))

        tmp_list = [t for t in f_words.most_common(24000) if t[1] >= 10]
        most_common = sorted(tmp_list, key=lambda tup: tup[1])
        most_common_words = [tup[0] for tup in most_common]

        final_words = set(most_common_words)
        word2index = {w: i for i, w in enumerate(final_words)}

        indices = OrderedDict()
        values = []

        context_windows_list = get_context_windows(sents)
        print(len(context_windows_list))

        for context_window in tqdm(context_windows_list):

            center_word = context_window[0]

            if center_word not in word2index:
                continue

            for j, context_word in enumerate(context_window):

                if (context_word in word2index) and (context_word != center_word): #and \
                        #((context_word in words_to_search) or (center_word in words_to_search)):
                    index_of_words = (word2index[center_word], word2index[context_word])
                    if index_of_words not in indices:
                        indices[index_of_words] = len(values)
                        values.append(0)
                    values[indices[index_of_words]] += 1/j

        torch_indices = torch.LongTensor(list(indices.keys()))
        torch_values = torch.tensor(values)

        with open(data_file, "wb") as f:
            pickle.dump((torch_indices, torch_values), f)

        with open(word2index_file, "wb") as f:
            pickle.dump(word2index, f)

    else:
        with open(data_file, "rb") as f:
            torch_indices, torch_values = pickle.load(f)

    print("Entries", torch_values.size())
    return torch_indices, torch_values


def get_glove_vectors_and_train_indices():
    with open(word2index_file, "rb") as f:
        word2index = pickle.load(f)

    sorted_w2i = sorted(word2index.items(), key=lambda tup: tup[1])
    sorted_wordlist = [tup[0] for tup in sorted_w2i]

    glove_vec_instance = GloveTweetVectors()
    w2v_dict = glove_vec_instance.get_word_to_vec_dict(sorted_wordlist)
    glove_vectors = [w2v_dict[word] for word in sorted_wordlist]
    for word in words_to_search:
        glove_vectors[word2index[word]] = np.random.normal(size=glove_vec_instance.zero_vector.shape)
    train_indices = np.array([word2index[word] for word in words_to_search], dtype=int)
    train_bool_indices = np.zeros(len(glove_vectors), dtype=bool)
    train_bool_indices[train_indices] = True

    return torch.tensor(glove_vectors), torch.LongTensor(train_bool_indices)


def remove_irony_hashtag(sentence):
    return sentence.replace("#irony", "").replace("#sarcasm", "")


def tweets_pre_processing(sentences: List[str]):
    """

    We remove URLs
    We remove usernames (starting with @)
    We remove #irony and #sarcasm
    We remove the hash and just keep the tag words
    All non-alphabetical characters except for 0 are removed

    Reference:
    https://www3.tuhh.de/sts/hoou/data-quality-explored/3-2-simple-transf.html

    :param sentences:
    :return: preprocessed sentences
    """

    def camel_case_split(identifier):
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
        return [m.group(0) for m in matches]

    result_sentences = []
    for i, sentence in tqdm(enumerate(sentences)):
        sentence = re.sub(r'https?://[^ ]+', '', sentence)
        sentence = re.sub(r'@[^ ]+', '', sentence)
        sentence = remove_irony_hashtag(sentence)
        sentence = re.sub(r'#', '', sentence)
        sentence = re.sub(r'([A-Za-z])\1{2,}', r'\1', sentence)
        sentence = re.sub(r' 0 ', 'zero', sentence)
        sentence = re.sub(r'[^A-Za-z ]', '', sentence)
        sentence = word_tokenize(sentence)
        sentence = [
            word for multi_word in sentence
            for word in camel_case_split(multi_word)
        ]
        if 10 <= len(sentence) <= 40:
            result_sentences.append(sentence)

    return result_sentences


if __name__ == "__main__":
    get_data()
