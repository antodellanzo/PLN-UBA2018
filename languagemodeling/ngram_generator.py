import random


class NGramGenerator(object):

    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self._n = model._n

        # compute the probabilities
        self._probs = self.getNgramProbabilities(model)

        # sort in descending order for efficient sampling
        sorted_probs = {}
        for key, value in self._probs.items():
            sorted_values = sorted(value.items(), key=lambda x: (-x[1], x[0]))
            sorted_probs[key] = sorted_values
        self._sorted_probs = sorted_probs

    def getNgramProbabilities(self, model):
        probs = dict()
        for ngram in model._count:
            if len(ngram) == self._n:
                prev_tokens = ngram[0:self._n-1]
                token = ngram[self._n-1]
                probs.setdefault(prev_tokens, dict())
                probs[prev_tokens][token] = model.cond_prob(token, prev_tokens)
        return probs

    def generate_sent(self):
        """Randomly generate a sentence."""
        n = self._n

        sent = []
        prev_tokens = ['<s>'] * (n - 1)
        if n == 1:
            token = self.generate_token()
        else:
            token = self.generate_token(tuple(prev_tokens))
        while token != '</s>':
            sent.append(token)
            prev_tokens.append(token)

            if n == 1:
                token = self.generate_token()
            else:
                prev_n_tkns_index = len(prev_tokens)-n+1
                prev_n_tokens = prev_tokens[prev_n_tkns_index:len(prev_tokens)]
                token = self.generate_token(tuple(prev_n_tokens))

        return sent

    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.

        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self._n
        if not prev_tokens:
            prev_tokens = ()
        assert len(prev_tokens) == n - 1

        probs = self._sorted_probs[prev_tokens]
        token = self.sample(probs)

        return token

    def sample(self, problist):
        r = random.random()  # entre 0 y 1
        i = 0
        word, prob = problist[0]
        acum = prob
        while r > acum and i < len(problist) - 1:
            i += 1
            word, prob = problist[i]
            acum += prob

        return word
