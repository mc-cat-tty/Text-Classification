#!/bin/python

import spacy
import logging
import pickle
import string
import re
import sys
import os
import math

BASIC = 0
LEMMATIZATION = 10
REMOVE_STOPWORDS = 20
LOW = 30
MEDIUM = 40
HIGH = 50

LOG_FILENAME = 'classification_tools.log'
LOG_LEVEL = logging.INFO
SPACY_MODEL = 'en_core_web_sm'
MAX_DIM = 1000000  # 1000 KB
NUM = 'NUM'

MODELS = os.path.join(os.getcwd(), 'models')
DATASET = os.path.join(os.getcwd(), 'dataset')
STOPWORDS_MODEL_FILENAME = os.path.join(MODELS, 'stopwords')

logging.basicConfig(filename=LOG_FILENAME, level=LOG_LEVEL, format='%(levelname)s | %(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', filemode='wt')
try:
    nlp = spacy.load(SPACY_MODEL, disable=['parser', 'ner'])  # Natural Language Processing
except:
    logging.error('Spacy has encountered problems while uploading "{}" model'.format(SPACY_MODEL))

punctuation = string.punctuation


class RawText(object):
    '''This class is meant to be a container for original text and some properties related to it'''

    def __init__(self, text='', fast=False, *args, **kwargs):
        '''
        :param text (string): If you want to upload the text from a text file, create the object with no text and then call read_text_from_file()
        :param fast (bool): True if you want to avoid useless computation (at the expense of less parameters)
        '''
        self.text = text
        self.rows_num = len(self.text.split("\n"))
        self.words_num = len(self.text.split())
        self.chars_num = len(self.text)
        if not fast:
            self.words_occurrences = self.__count_words_occurrences()
            self.words_freq = self.__get_words_freq()
        logging.info('RawText object initialized using text starting with "{}..."'.format(self.text[:5]))

    def __count_words_occurrences(self):
        return self.count_words_occurrences(self.text)

    @staticmethod
    def count_words_occurrences(text):
        logging.info('Counting words occurrences')
        text = text.split()
        s = set(text)
        return {word: text.count(word) for word in s}

    def __get_words_freq(self):
        return self.get_words_freq(self.words_occurrences, self.words_num)

    @staticmethod
    def get_words_freq(words_occurrences, words_num):
        logging.info('Calculating words frequencies with occurrences normalized')
        return {word: words_occurrences[word] / words_num for word in
                words_occurrences.keys()}  # Normalizing occurrences (number between 0 and 1)

    @classmethod
    def from_text_file(cls, filename, *args, **kwargs):
        try:
            with open(filename, 'rt') as file:
                logging.info('File "{}" opened'.format(filename))
                text = file.read()
                size = len(text)
                logging.info('File "{}" read'.format(filename))
                logging.debug('Size: {}'.format(size))
                if size > MAX_DIM:
                    text = text[:MAX_DIM]
                    logging.warning("File too big. It has been reduced")
        except:
            logging.error('File "{}" could not be opened'.format(filename))
        else:
            return cls(text, *args, **kwargs)


class CleanedText(RawText):
    '''This class contains cleaned text'''

    def __init__(self, text='', cleaning_level=HIGH, *args, **kwargs):
        '''
        :param cleaning_level (int): BASIC, LEMMATIZATION or REMOVE_STOPWORDS; LOW, MEDIUM and HIGH. Default level is HIGH.
            BASIC level removes only punctuation and numbers; it also transforms text in lowercase
            LEMMATIZATION level lemmatizes all text's words
            REMOVE_STOPWORDS level removes stopwords and worthless words (see tf-idf)
            LOW level is the same as BASIC
            MEDIUM level is equivalent to BASIC + LEMMATIZATION
            HIGH level is identical to BASIC + LEMMATIZATION + REMOVE_STOPWORDS
        :param text (string): If you want to upload the text from a text file, create the object with no text and then call read_text_from_file()
        :param stopwords_model (Stopwords): Pass this paramter only if the cleaning_level is HIGH or REMOVE_STOPWORDS
        '''
        super().__init__(text, *args, **kwargs)
        self.cleaning_level = cleaning_level
        self.cleaned_text = self.__clean_text()
        logging.info('CleanedText object initialized using "{}" as cleaning level'.format(self.cleaning_level))
        del self.text
        logging.info("Cleaning up useless attributes to free memory")

    def __clean_text(self):
        return self.clean_text(self.text, self.cleaning_level)

    @staticmethod
    def clean_text(text, cleaning_level):  # Preprocessing
        logging.info('Cleaning text at level "{}"'.format(cleaning_level))
        if cleaning_level == BASIC or cleaning_level == LOW:
            return CleanedText.__basic_cleaning(text)
        elif cleaning_level == LEMMATIZATION:
            return CleanedText.__lemmatization(text)
        elif cleaning_level == REMOVE_STOPWORDS:
            return CleanedText.__remove_stopwords(text)
        elif cleaning_level == MEDIUM:
            return CleanedText.__lemmatization(CleanedText.__basic_cleaning(text))
        elif cleaning_level == HIGH:
            return CleanedText.__remove_stopwords(CleanedText.__lemmatization(CleanedText.__basic_cleaning(text)))

    @staticmethod
    def __basic_cleaning(text):
        return re.sub(r'\s+', ' ',
                      re.sub(r'\d+', NUM,
                             text.lower().strip().translate(str.maketrans(punctuation, ' ' * len(punctuation)))))

    @staticmethod
    def __lemmatization(text):  # Tokenization and Lemmatization
        try:
            nlp_str = nlp(text)
        except:
            logging.critical("Spacy model hasn't been loaded")
            sys.exit()
        else:
            return ' '.join([token.lemma_ for token in nlp_str])

    @staticmethod
    def __remove_stopwords(text):
        return Stopwords.remove_stopwords_from(text)


class AbstractText(CleanedText):
    '''
    This class is an abstract representation of the original text - unstructured data
    '''

    def __init__(self, text='', cleaning_level=HIGH, fast=False, *args, **kwargs):
        '''
        :param text (string): If you want to upload the text from a text file, create the object with no text and then call read_text_from_file()
        :param fast (bool): True if you want to avoid useless computation (at the expense of less parameters)
        '''
        super().__init__(text, cleaning_level, fast, *args, **kwargs)
        self.words_num = len(self.cleaned_text.split())  # Overriding old properties
        if not fast:
            self.chars_num = len(self.cleaned_text.replace(' ', ''))  # Count characters excluding whitespaces
        self.words_occurrences = self.count_words_occurrences(self.cleaned_text)
        self.words_freq = self.get_words_freq(self.words_occurrences, self.words_num)
        self.words_set = set(self.cleaned_text.split())
        logging.info('AbstractText object initialized')
        del self.cleaned_text  # Deleting every reference to the text
        logging.info("Cleaning up useless attributes to free memory")


class Model(object):  # TODO: loggging
    '''This is a class that implements methods and attributes for a generic model'''

    def __init__(self, *args, **kwargs):
        '''
        :param --> Pass to this function an arbitrary number of abstract_text to create a model
        '''
        self.sets_list = list()
        self.intersection = None
        self.intersection_dim = None
        self.union = None
        self.union_dim = None
        self.difference = None
        self.difference_dim = None
        self.words_num = 0
        self.chars_num = 0
        self.words_occurrences = dict()
        self.words_freq = None
        if len(args) != 0:
            self.add_abstract_text(*args)

    @staticmethod
    def get_intersection(sets_list):
        return set.intersection(*sets_list)

    @staticmethod
    def get_difference(union, intersection):
        return union - intersection

    @staticmethod
    def get_union(sets_list):
        return set.union(*sets_list)

    @staticmethod
    def total_words_num(*args):
        return sum([abstract_text.words_num for abstract_text in args])

    @staticmethod
    def total_chars_num(*args):
        return sum([abstract_text.chars_num for abstract_text in args])

    @staticmethod
    def merge_words_occurrences_dicts(*args):
        d = dict()
        for abstract_text in args:
            for word, occurrences in abstract_text.words_occurrences.items():
                if word in d:  # Current word already present
                    d[word] += occurrences
                else:
                    d.setdefault(word, occurrences)
        return d

    def __update_words_occurrences_dict(self, *args):
        d = self.words_occurrences
        for abstract_text in args:
            for word, occurrences in abstract_text.words_occurrences.items():
                if word in d:  # Current word already present
                    d[word] += occurrences
                else:
                    d.setdefault(word, occurrences)
        return d

    def __update_attributes(self, *args):
        self.intersection = self.get_intersection(self.sets_list)
        self.intersection_dim = len(self.intersection)
        self.union = self.get_union(self.sets_list)
        self.union_dim = len(self.union)
        self.difference = self.get_difference(self.union, self.intersection)
        self.difference_dim = len(self.difference)
        self.words_num += self.total_words_num(*args)
        self.chars_num += self.total_chars_num(*args)
        self.words_occurrences = self.__update_words_occurrences_dict(*args)
        self.words_freq = RawText.get_words_freq(self.words_occurrences, self.words_num)

    def add_abstract_text(self, *args):
        self.sets_list.extend([abstract_text.words_set for abstract_text in args])
        self.__update_attributes(*args)

    def save(self, filename):
        try:
            with open(filename, 'wb') as filehandler:
                pickle.dump(self, filehandler)
        except:
            logging.error('File "{}" could not be opened'.format(filename))

    @staticmethod
    def load(filename):
        try:
            with open(filename, 'rb') as filehandler:
                return pickle.load(filehandler)
        except:
            logging.error('File "{}" could not be opened'.format(filename))


class Stopwords(Model):  # TODO: logging and docstrings
    THRESHOLD = 0.01  # Frequency value above which (popular) words are considered stopwords
    stopwords_list = list()
    stopwords_regex = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stopwords_list = None
        if len(args) != 0:
            self.__populate_stopwords_list()

    def __update_class_stopwords_list(self):
        Stopwords.stopwords_list = self.stopwords_list
        regex = list(r'\b(')
        for word in self.stopwords_list:
            regex.append('{}|'.format(word))
        regex.remove(NUM + '|')
        regex.append(r')\b')
        regex = ''.join(regex)
        Stopwords.stopwords_regex = re.compile(regex)

    def __populate_stopwords_list(self):
        self.stopwords_list = list()
        for word, freq in self.words_freq.items():
            if freq > self.THRESHOLD:
                self.stopwords_list.append(word)
        self.__update_class_stopwords_list()

    def update_stopwords_list(self, threshold):  # Recalculate stopwords list using different threshold
        Stopwords.THRESHOLD = threshold
        self.__populate_stopwords_list()

    @staticmethod
    def is_present(word):
        return word in Stopwords.stopwords_list

    @staticmethod
    def remove_stopwords_from(text):
        buffer = list()
        for word in text.split():
            if word not in Stopwords.stopwords_list or word == NUM:
                buffer.append(word)
        return ' '.join(buffer)

    def add_and_train(self, *args):  # Add some AbstractText to train more the model
        self.add(*args)
        self.train()  # Update stopwords list

    def add(self, *args):
        self.add_abstract_text(*args)

    def train(self):  # Extract stopwords from model
        self.__populate_stopwords_list()  # Update stopwords list


class Vocabulary(Model):
    def __init__(self, label, stopwords_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label
        self.words_balanced = None  # Words importance (weight) calculated using TF-IDF (Term Frequency - Inverse Document Frequency)
        self.doc_collection = stopwords_model.sets_list  # Stopwords is used as a representation of english language words' distribution
        self.doc_num = len(self.doc_collection)  # Number of documents in the corpus (collection)
        if len(args) != 0:
            self.__populate_words_balanced()

    def __eq__(self, other):
        if not isinstance(other, Vocabulary):
            return False

        return self.label == other.label

    def tf_idf(self, word, tf):
        '''
        :param tf: Term Freqeuncy in the current document
        :return: TF-IDF coefficient
        '''

        d = 1  # Number of documents in the corpus that contain the word (it starts form 1 to avoid division-by-zero)
        for doc in self.doc_collection:
            if word in doc:
                d += 1

        idf = math.log((self.doc_num + 1) / d,
                       10)  # doc_num: Number of documents in the corpus (collection) --> +1 to Compensate d's adjustment

        return tf * idf

    def __populate_words_balanced(self):
        self.words_balanced = dict()
        for word in self.union:
            self.words_balanced[word] = self.tf_idf(word, self.words_freq[word])

    def add_and_train(self, *args):
        self.add(*args)
        self.train()

    def add(self, *args):
        self.add_abstract_text(*args)

    def train(self):
        self.__populate_words_balanced()

    def change_stopwords_model(self, stopwords_model):
        self.doc_collection = stopwords_model.sets_list
        self.doc_num = len(self.doc_collection)
        self.__populate_words_balanced()

    def compare(self, abstract_text):
        score = 0
        for word, weight in self.words_balanced.items():
            if word in abstract_text.words_set:
                score += weight * abstract_text.words_freq[word]
        return score


class Classificator(object):
    '''This class is a collection of useful tools'''

    @staticmethod
    def train_stopwords_and_save_model(dataset_directory, model_filename):
        logging.info('Starting stopwords training')
        s = Stopwords()
        for filename in os.listdir(dataset_directory):
            a = AbstractText.from_text_file(os.path.join(dataset_directory, filename), cleaning_level=MEDIUM, fast=True)
            s.add(a)
        s.train()
        s.save(model_filename)
        logging.info('Stopwords training finished')
        return s

    @staticmethod
    def train_stopwords_starting_from_model_and_save_new_model(old_model, dataset_directory, new_model_filename):
        logging.info('Starting stopwords training')
        s = old_model
        for filename in os.listdir(dataset_directory):
            a = AbstractText.from_text_file(os.path.join(dataset_directory, filename), cleaning_level=MEDIUM, fast=True)
            s.add(a)
        s.train()
        s.save(new_model_filename)
        logging.info('Stopwords training finished')
        return s

    @staticmethod
    def train_vocabulary_and_save_model(label, stopwords_model, dataset_directory, model_filename):
        logging.info('Starting {} vocabulary training'.format(label))
        v = Vocabulary(label, stopwords_model)
        for filename in os.listdir(dataset_directory):
            a = AbstractText.from_text_file(os.path.join(dataset_directory, filename), cleaning_level=HIGH, fast=True)
            v.add(a)
        v.train()
        v.save(model_filename)
        logging.info('Training finished for {} vocabulary'.format(label))
        return v

    @staticmethod
    def train_vocabulary_starting_from_model_and_save_new_model(old_model, dataset_directory, new_model_filename):
        logging.info('Starting {} vocabulary training'.format(old_model.label))
        v = old_model
        for filename in os.listdir(dataset_directory):
            a = AbstractText.from_text_file(os.path.join(dataset_directory, filename), cleaning_level=HIGH, fast=True)
            v.add(a)
        v.train()
        v.save(new_model_filename)
        logging.info('Training finished for {} vocabulary'.format(old_model.label))
        return v

    @staticmethod
    def init_stopwords(model_filename):
        try:
            stopwords_model = Stopwords.load(model_filename)  # Loading stopwords model
        except:
            logging.error(
                'Stopwords has encountered problems while uploading "{}" model'.format(model_filename))
            sys.exit()
        else:
            stopwords_model.update_stopwords_list(
                0.003)  # Updating threshold and regenerating stopwords dictionary (this value deletes nonsignificant words
        return stopwords_model

    @staticmethod
    def init_stopwords_default():
        Classificator.init_stopwords(STOPWORDS_MODEL_FILENAME)  # Initializing stopwords class


class LabelledText(AbstractText):
    def __init__(self, text='', vocabularies_list=list(), *args, **kwargs):
        super().__init__(text, *args, **kwargs)
        self.vocabularies_list = vocabularies_list
        self.label = str()
        if vocabularies_list != list():
            self.__update_label()
        else:
            self.updated = True

    def add_vocabulary(self, *args):
        if args != tuple():
            self.vocabularies_list.extend(args)
            self.updated = False

    def get_label(self):
        if not self.updated:
            self.__update_label()
        return self.label

    def __update_label(self):
        d = {v.compare(self): v.label for v in self.vocabularies_list}
        self.label = d.get(max(d.keys()))
        self.updated = True


def main():  # Test function
    # Classificator.train_stopwords_and_save_model(os.path.join(DATASET, 'Canadian_Parliament_Debates'),
    #                                              os.path.join(MODELS, 'stopwords_old'))  # Instruction to train new stopwords model
    # s = Stopwords.load(os.path.join(MODELS, 'stopwords_old'))
    # Classificator.train_stopwords_starting_from_model_and_save_new_model(s, os.path.join(DATASET, "Joy"),
    #                                                                      os.path.join(MODELS, 'stopwords'))  # Instruction to reinforce already existing model
    # s = Classificator.init_stopwords(STOPWORDS_MODEL_FILENAME)
    # Classificator.train_vocabulary_and_save_model("happiness", s, os.path.join(DATASET, 'Happiness/'),
    #                                               os.path.join(MODELS, 'happiness_vocabulary'))
    # Classificator.train_stopwords_starting_from_model_and_save_new_model(s, os.path.join(DATASET, "Reviews/"),
    #                                                                      os.path.join(MODELS, 'stopwords'))  # Instruction to reinforce already existing model
    # Classificator.train_vocabulary_and_save_model('sadness', s, os.path.join(DATASET, 'Sadness/'), os.path.join(MODELS, 'sadness_vocabulary'))

    Classificator.init_stopwords(STOPWORDS_MODEL_FILENAME)  # Initializing stopwords class

    happiness_vocabulary = Vocabulary.load(os.path.join(MODELS, 'happiness_vocabulary'))
    sadness_vocabulary = Vocabulary.load(os.path.join(MODELS, 'sadness_vocabulary'))

    l = LabelledText("Today is a gorgeous day, the sun is shining and the sky is blue.",
                     [happiness_vocabulary, sadness_vocabulary], cleaning_level=HIGH, fast=True)
    print(l.get_label())

    l = LabelledText("Nobody cares for me, I'm worthless.", [happiness_vocabulary, sadness_vocabulary],
                     cleaning_level=HIGH, fast=True)
    print(l.get_label())




if __name__ == "__main__":
    main()
