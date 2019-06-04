from typing import List, Hashable
from scipy.sparse import csr_matrix

from . import libtfidf


__all__ = ['tfidf', 'tf', 'Vocabulary', 'Corpus']


class Vocabulary:
    """Memoize the all words in the corpus"""

    def __init__(self):
        self.words = set()
        self.i2w = []
        self.w2i = {}

    def __len__(self):
        return len(self.words)

    def __contains__(self, word):
        return word in self.words

    def add(self, word):
        """add new word"""
        if word in self.words:
            return
        self.i2w.append(word)
        self.w2i[word] = len(self.words)
        self.words.add(word)

    def get(self, idx):
        """get a word by index"""
        return self.i2w[idx]

    def index(self, word):
        """get its index by word"""
        return self.w2i[word]


class Corpus:
    """Make a corpus"""
    def __init__(self, docs: List[List[Hashable]]):
        """Constructor

        Parameters
        ----------
        docs: List[List[Hashable]]
            a list of lists of words
            words are any (hashable) values
        """
        voc = Vocabulary()
        for doc in docs:
            for w in doc:
                voc.add(w)
        docs_int = [
            [voc.index(w) for w in doc]
            for doc in docs
        ]
        self.voc = voc
        self.docs = docs_int


def _to_csr_matrix(_csr_matrix) -> csr_matrix:
    n, m, data = _csr_matrix
    rows = [item[0] for item in data]
    cols = [item[1] for item in data]
    vals = [item[2] for item in data]
    return csr_matrix((vals, (rows, cols)), shape=(n, m))


def tfidf(corpus: Corpus) -> csr_matrix:
    """Compute tf-idf Matrix for docs

    Parameters
    ----------
    corpus : Corpus
    """
    return _to_csr_matrix(libtfidf.tfidf(corpus.docs))


def tf(corpus: Corpus) -> csr_matrix:
    """Compute tf Matrix (just word count) for docs

    Parameters
    ----------
    corpus : Corpus
    """
    return _to_csr_matrix(libtfidf.tf(corpus.docs))
