"""Evaulate a language model using a test set.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Language model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle
import math

from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import RegexpTokenizer
from languagemodeling.ngram import NGram, AddOneNGram, InterpolatedNGram

pattern = r'''(?x)    # set flag to allow verbose regexps
   (?:\d{1,3}(?:\.\d{3})+)  # numbers with '.' in the middle
   | (?:[Ss]r\.|[Ss]ra\.|art\.)  # common spanish abbreviations
   | (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
   | \w+(?:-\w+)*        # words with optional internal hyphens
   | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
   | \.\.\.            # ellipsis
   | [][.,;"'?():-_`]  # these are separate tokens;
'''

if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the model
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    tokenizer = RegexpTokenizer(pattern)
    corpus = PlaintextCorpusReader('.', 'test_corpus.txt', word_tokenizer=tokenizer)
    sents = corpus.sents()

    # compute the cross entropy
    log_prob = model.log_prob(sents)
    n = sum(len(sent) + 1 for sent in sents)  # count '</s>' event
    e = - log_prob / n
    p = math.pow(2.0, e)

    print('Log probability: {}'.format(log_prob))
    print('Cross entropy: {}'.format(e))
    print('Perplexity: {}'.format(p))
