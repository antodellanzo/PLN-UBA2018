# https://docs.python.org/3/library/collections.html
from collections import defaultdict
import copy
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
        return -float('inf')

    def log_prob(self, sents):
        result = 0.0
        for i, sent in enumerate(sents):
            lp = self.sent_log_prob(sent)
            if lp == -float('inf'):
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
            new_sentence = copy.deepcopy(sent)
            self.addDelimiterToSentence(new_sentence, self._n)
            self.updateCountOfSentenceWithNgram(count, new_sentence, n)
            self.updateCountOfSentenceWithNgram(count, new_sentence, n - 1)

        self._count = dict(count)

    def updateCountOfSentenceWithNgram(self, count, sent, n):
        maximumStartOfNgram = len(sent) - (n - 1)
        for i in range(min(maximumStartOfNgram, len(sent))):
            ngram = tuple(sent[i:i+n])
            count[ngram] += 1

    def addDelimiterToSentence(self, sent, n):
        for i in range(n - 1):
            sent.insert(0, '<s>')
        sent.append('</s>')

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.

        tokens -- the n-gram or (n-1)-gram tuple.
        """
        return self._count.get(tuple(tokens), 0)

    def cond_prob(self, token, prev_tokens=tuple()):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        return self.cond_prob_ngram(token, prev_tokens)

    def cond_prob_ngram(self, token, prev_tokens=tuple()):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        if self.count(prev_tokens) == 0:
            return 0
        else:
            if isinstance(prev_tokens, list):
                whole_sentence_as_list = prev_tokens + [token]
            else:
                whole_sentence_as_list = list(prev_tokens + (token,))
            return self.count(whole_sentence_as_list) / self.count(prev_tokens)

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        self.addDelimiterToSentence(sent, self._n)
        prob = 1

        for i in range(len(sent) - self._n + 1):
            ngramSent = sent[i:i+self._n]
            prob *= self.cond_prob_of_ngram_sentence(ngramSent)

        return prob

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        self.addDelimiterToSentence(sent, self._n)
        prob = 0

        for i in range(len(sent) - self._n + 1):
            ngramSent = sent[i:i+self._n]
            cond_prob = self.cond_prob_of_ngram_sentence(ngramSent)

            if cond_prob != 0:
                prob += math.log(cond_prob, 2)
            else:
                return -float('inf')

        return prob

    def cond_prob_of_ngram_sentence(self, sentence):
        assert len(sentence) >= self._n

        if (self._n > 1):
            return self.cond_prob(sentence[self._n-1], sentence[0:self._n-1])
        else:
            return self.cond_prob(sentence[self._n-1])


class AddOneNGram(NGram):

    def __init__(self, n, sents):
        vocabulary = set()
        vocabulary.add("<\s>")

        for sent in sents:
            for word in sent:
                vocabulary.add(word)
        self._vocabularySize = len(vocabulary)

        super().__init__(n, sents)

    def V(self):
        """Size of the vocabulary.
        """
        return self._vocabularySize

    def cond_prob(self, token, prev_tokens=[]):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        return self.cond_prob_add_one_ngram(token, prev_tokens)

    def cond_prob_add_one_ngram(self, token, prev_tokens=[]):
        if isinstance(prev_tokens, list):
            whole_sentence_as_list = prev_tokens + [token]
        else:
            whole_sentence_as_list = list(prev_tokens + (token,))
        sentence_prob = self.count(whole_sentence_as_list)
        return (sentence_prob + 1) / (self.count(prev_tokens) + self.V())


class InterpolatedNGram(AddOneNGram):

    def __init__(self, n, sents, gamma=None, addone=True):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        gamma -- interpolation hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        # save 10% of data to estimate gamma if no gamma was given
        # use 90% of data for training
        if gamma is None:
            total_sents = len(sents)
            held_out_data_size = int(90.0 * total_sents / 100.0)
            held_out_data = sents[held_out_data_size:]
            sents = sents[0:held_out_data_size]

        self._gamma = gamma
        self._should_use_add_one = addone

        super().__init__(n, sents)

        count = defaultdict(int)

        # store missing m-grams with m < n-1
        # ((n-1)-grams and n-grams where stored in super constructor)
        for sent in sents:
            for i in range(0, n-1):
                sent_copy = copy.deepcopy(sent)
                self.addDelimiterToSentence(sent_copy, i)
                self.updateCountOfSentenceWithNgram(count, sent_copy, i)

        # update _count attribute with this new data
        for key, value in count.items():
            self._count[key] = value

        # estimate gamma using held-out data if no gamma was given
        # select gamma that minimizes perplexity
        if gamma is None and held_out_data is not None:
            gammas = [1.0, 5.0, 10.0, 50.0, 100.0]
            min_perplexity = float('inf')
            gamma_of_min_perplexity = -1
            for current_gamma in gammas:
                self._gamma = current_gamma
                perplexity = self.perplexity(held_out_data)
                if perplexity < min_perplexity:
                    min_perplexity = perplexity
                    gamma_of_min_perplexity = current_gamma
            self._gamma = gamma_of_min_perplexity

    def cond_prob(self, token, prev_tokens=[]):
        cond_probs = list()
        prev_n = self._n - 1

        # get cond_prob of token for m-grams with 1<m<n
        for i in range(0, prev_n):
            cond_prob = self.cond_prob_ngram(token, prev_tokens[i:])
            cond_probs.append(cond_prob)
        # check if should use add one cond prob for unigram
        if self._should_use_add_one:
            last_cond = self.cond_prob_add_one_ngram(token)
        else:
            last_cond = self.cond_prob_ngram(token)
        cond_probs.append(last_cond)

        # alculate lambdas
        lambdas = list()
        for i in range(0, prev_n):
            prev_tokens_count = self.count(prev_tokens[i:])
            numerator = (1-sum(lambdas)) * prev_tokens_count
            denominator = prev_tokens_count + self._gamma
            current_lambda = numerator / denominator
            lambdas.append(current_lambda)
        last_lambda = 1
        for i in range(0, len(lambdas)):
            last_lambda -= lambdas[i]
        lambdas.append(last_lambda)

        # calculate final cond prob
        prob = 0
        for i in range(0, len(cond_probs)):
            prob += lambdas[i] * cond_probs[i]

        return prob
