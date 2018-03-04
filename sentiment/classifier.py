from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from nltk import word_tokenize
from nltk.corpus import stopwords
import re

classifiers = {
    'maxent': LogisticRegression,
    'mnb': MultinomialNB,
    'svm': LinearSVC,
}


class SentimentClassifier(object):

    def __init__(self, clf='svm', optimize_vect=False, binary_counts=False, filter_stop_words=False):
        """
        clf -- classifying model, one of 'svm', 'maxent', 'mnb' (default: 'svm').
        """
        self._clf = clf
        self._pipeline = pipeline = Pipeline([
            ('vect', self.tokenizer(optimize_vect, binary_counts, filter_stop_words)),
            ('clf', classifiers[clf]()),
        ])

    def replace_occurrences_of_vowel(self, vowel, word, words_set):
        regex = vowel + '{3,}'
        regexp = re.compile(regex)
        if regexp.search(word):
            word_one_vowel = re.sub(regex, vowel, word)
            word_two_vowel = re.sub(regex, vowel+vowel, word)
            if word_two_vowel in words_set:
                return word_two_vowel
            else:
                return word_one_vowel
        else:
            return word

    def negate_tweet(self, tweet):
        tokens = tweet.split()
        new_tokens = []
        negate = False
        for token in tokens:
            if token in ['no', 'ni', 'tampoco', 'nunca', 'jamás', 'nada']:
                negate = True
            elif token in ['.', ',', ';', ':', '?', '!', ')', '"', '-', ']']:
                negate = False
            elif negate:
                token = 'NOT_' + token
            new_tokens.append(token)
        return ' '.join(new_tokens)

    def fit(self, X, y, normalize=False, negate=False):
        X_list = list(X)
        new_tweet_list = list()
        if normalize:
            words = set()
            mentions = r'(?:@[^\s]+)'
            urls = r'(?:https?\://t.co/[\w]+)'
            for tweet in X_list:
                for word in tweet:
                    words.add(word)
            for tweet in X_list:
                tweet_without_mentions = re.sub(mentions, '', tweet)
                tweet_without_urls = re.sub(urls, '', tweet_without_mentions)
                final_tweet = tweet_without_urls
                for vowel in {'a', 'e', 'i', 'o', 'u'}:
                    final_tweet = self.replace_occurrences_of_vowel(vowel, final_tweet, words)
                new_tweet_list.append(final_tweet)
        if negate:
            for tweet in X_list:
                new_tweet_list.append(self.negate_tweet(tweet))
        if not new_tweet_list:
            new_tweet_list = X_list
        self._pipeline.fit(new_tweet_list, y)

    def predict(self, X):
        return self._pipeline.predict(X)

    def tokenizer(self, custom_tokenizer=False, ingore_word_repetition=False, stop_words=False):
        if stop_words:
            stop_words_set = set(stopwords.words('spanish'))
        else:
            stop_words_set = set()
        if custom_tokenizer:
            return CountVectorizer(tokenizer=word_tokenize, binary=ingore_word_repetition, stop_words=stop_words_set)
        else:
            return CountVectorizer(binary=ingore_word_repetition, stop_words=stop_words_set)

