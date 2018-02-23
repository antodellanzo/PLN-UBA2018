from collections import defaultdict


class BadBaselineTagger:

    def __init__(self, tagged_sents):
        """
        tagged_sents -- training sentences, each one being a list of pairs.
        """
        pass

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        return [self.tag_word(w) for w in sent]

    def tag_word(self, w):
        """Tag a word.

        w -- the word.
        """
        return 'nc0s000'

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return True


class BaselineTagger:

    def __init__(self, tagged_sents):
        """
        tagged_sents -- training sentences, each one being a list of pairs.
        """
        word_tags_count = dict()
        tag_counts = defaultdict(int)
        for tagged_sent in tagged_sents:
            for word_and_tag in tagged_sent:
                word = word_and_tag[0]
                tag = word_and_tag[1]
                tag_counts[tag] += 1
                word_tags_count.setdefault(word, defaultdict(int))
                word_tags_count[word][tag] += 1

        self._most_common_tag = sorted(tag_counts.items(), key=lambda x: -x[1])[0][0]
        self._word_tags_count = dict()
        for word, tags_dict in word_tags_count.items():
            self._word_tags_count[word] = sorted(tags_dict.items(), key=lambda x: -x[1])

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        return [self.tag_word(w) for w in sent]

    def tag_word(self, w):
        """Tag a word.

        w -- the word.
        """
        if self.unknown(w):
            return self._most_common_tag
        return self._word_tags_count.get(w)[0][0]

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return w not in self._word_tags_count.keys()
