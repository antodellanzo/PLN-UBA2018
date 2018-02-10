# https://docs.python.org/3/library/collections.html
from collections import defaultdict
import math


class LanguageModel(object):

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        return 0.0

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        return -math.inf

    def log_prob(self, sents):
        result = 0.0
        for i, sent in enumerate(sents):
            lp = self.sent_log_prob(sent)
            if lp == -math.inf:
                return lp
            result += lp
        return result

    def cross_entropy(self, sents):
        log_prob = self.log_prob(sents)
        n = sum(len(sent) + 1 for sent in sents)  # count '</s>' events
        e = - log_prob / n
        return e

    def perplexity(self, sents):
        return math.pow(2.0, self.cross_entropy(sents))


class NGram(LanguageModel):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self._n = n

        count = defaultdict(int)

        for sent in sents:
            self.addDelimiterToSentence(sent)
            self.updateCountOfSentenceWithNgram( count, sent, n )
            self.updateCountOfSentenceWithNgram( count, sent, n - 1 )

        self._count = dict(count)

    def updateCountOfSentenceWithNgram(self, count, sent, n):
        maximumStartOfNgram = len(sent) - ( n - 1)
        for i in range( min( maximumStartOfNgram, len(sent) ) ):
            ngram = tuple( sent[i:i+n] )
            count[ngram] += 1

    def addDelimiterToSentence(self, sent):
        for i in range( self._n - 1 ):
            sent.insert(0, '<s>')
        sent.append('</s>')

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.

        tokens -- the n-gram or (n-1)-gram tuple.
        """
        return self._count.get( tuple(tokens), 0 )

    def cond_prob(self, token, prev_tokens=[]):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        if self.count(prev_tokens) == 0:
            return 0
        else:
            whole_sentence_as_list = ( prev_tokens + [token] ) if ( isinstance(prev_tokens, list) ) else list( prev_tokens + (token,) )
            return self.count( whole_sentence_as_list ) / self.count( prev_tokens )

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        self.addDelimiterToSentence(sent)
        prob = 1

        for i in range(len(sent) - self._n + 1):
            ngramSent = sent[i:i+self._n]
            prob *= self.cond_prob_of_sentence(ngramSent)
            
        return prob

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        self.addDelimiterToSentence(sent)
        prob = 0

        for i in range(len(sent) - self._n + 1):
            ngramSent = sent[i:i+self._n]
            cond_prob = self.cond_prob_of_sentence(ngramSent)

            if cond_prob != 0:
                prob += math.log( cond_prob, 2)
            else:
                return -float('inf')

        return prob

    def cond_prob_of_sentence(self, sentence):
        assert len(sentence) >= self._n

        if (self._n > 1):
            return self.cond_prob( sentence[self._n-1], sentence[0:self._n-1] )
        else:
            return self.cond_prob( sentence[self._n-1] )

