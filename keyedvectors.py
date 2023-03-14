import logging
import sys
import itertools
import warnings
from numbers import Integral
from typing import Iterable
from numpy import (
    dot, float32 as REAL, double, zeros, vstack, ndarray,
    sum as np_sum, prod, argmax, dtype, ascontiguousarray, frombuffer,
)
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist

import utils, matutils
from dictionary import Dictionary
from utils import deprecated


logger = logging.getLogger(__name__)


_KEY_TYPES = (str, int, np.integer)

_EXTENDED_KEY_TYPES = (str, int, np.integer, np.ndarray)


def _ensure_list(value):
    if value is None:
        return []

    if isinstance(value, _KEY_TYPES) or (isinstance(value, ndarray) and len(value.shape) == 1):
        return [value]

    if isinstance(value, ndarray) and len(value.shape) == 2:
        return list(value)

    return value


class KeyedVectors(utils.SaveLoad):

    def __init__(self, vector_size, count=0, dtype=np.float32, mapfile_path=None):
        self.vector_size = vector_size
        self.index_to_key = [None] * count
        self.next_index = 0
        self.key_to_index = {}
        self.vectors = zeros((count, vector_size), dtype=dtype)
        self.norms = None
        self.expandos = {}
        self.mapfile_path = mapfile_path

    def __str__(self):
        return f"{self.__class__.__name__}<vector_size={self.vector_size}, {len(self)} keys>"

    def _load_specials(self, *args, **kwargs):
        super(KeyedVectors, self)._load_specials(*args, **kwargs)
        if hasattr(self, 'doctags'):
            self._upconvert_old_d2vkv()
        if not hasattr(self, 'index_to_key'):
            self.index_to_key = self.__dict__.pop('index2word', self.__dict__.pop('index2entity', None))
        if not hasattr(self, 'vectors'):
            self.vectors = self.__dict__.pop('syn0', None)
            self.vector_size = self.vectors.shape[1]
        if not hasattr(self, 'norms'):
            self.norms = None
        if not hasattr(self, 'expandos'):
            self.expandos = {}
        if 'key_to_index' not in self.__dict__:
            self._upconvert_old_vocab()
        if not hasattr(self, 'next_index'):
            self.next_index = len(self)

    def _upconvert_old_vocab(self):
        old_vocab = self.__dict__.pop('vocab', None)
        self.key_to_index = {}
        for k in old_vocab.keys():
            old_v = old_vocab[k]
            self.key_to_index[k] = old_v.index
            for attr in old_v.__dict__.keys():
                self.set_vecattr(old_v.index, attr, old_v.__dict__[attr])
        if 'sample_int' in self.expandos:
            self.expandos['sample_int'] = self.expandos['sample_int'].astype(np.uint32)

    def allocate_vecattrs(self, attrs=None, types=None):
        if attrs is None:
            attrs = list(self.expandos.keys())
            types = [self.expandos[attr].dtype for attr in attrs]
        target_size = len(self.index_to_key)
        for attr, t in zip(attrs, types):
            if t is int:
                t = np.int64
            if t is str:
                t = object
            if attr not in self.expandos:
                self.expandos[attr] = np.zeros(target_size, dtype=t)
                continue
            prev_expando = self.expandos[attr]
            if not np.issubdtype(t, prev_expando.dtype):
                raise TypeError(
                    f"Can't allocate type {t} for attribute {attr}, "
                    f"conflicts with its existing type {prev_expando.dtype}"
                )
            if len(prev_expando) == target_size:
                continue
            prev_count = len(prev_expando)
            self.expandos[attr] = np.zeros(target_size, dtype=prev_expando.dtype)
            self.expandos[attr][: min(prev_count, target_size), ] = prev_expando[: min(prev_count, target_size), ]

    def set_vecattr(self, key, attr, val):
        self.allocate_vecattrs(attrs=[attr], types=[type(val)])
        index = self.get_index(key)
        self.expandos[attr][index] = val

    def get_vecattr(self, key, attr):
        index = self.get_index(key)
        return self.expandos[attr][index]

    def resize_vectors(self, seed=0):
        target_shape = (len(self.index_to_key), self.vector_size)
        self.vectors = prep_vectors(target_shape, prior_vectors=self.vectors, seed=seed)
        self.allocate_vecattrs()
        self.norms = None

    def __len__(self):
        return len(self.index_to_key)

    def __getitem__(self, key_or_keys):
        if isinstance(key_or_keys, _KEY_TYPES):
            return self.get_vector(key_or_keys)

        return vstack([self.get_vector(key) for key in key_or_keys])

    def get_index(self, key, default=None):
        val = self.key_to_index.get(key, -1)
        if val >= 0:
            return val
        elif isinstance(key, (int, np.integer)) and 0 <= key < len(self.index_to_key):
            return key
        elif default is not None:
            return default
        else:
            raise KeyError(f"Key '{key}' not present")

    def get_vector(self, key, norm=False):
        index = self.get_index(key)
        if norm:
            self.fill_norms()
            result = self.vectors[index] / self.norms[index]
        else:
            result = self.vectors[index]

        result.setflags(write=False)
        return result

    @deprecated("Use get_vector instead")
    def word_vec(self, *args, **kwargs):
        return self.get_vector(*args, **kwargs)

    def get_mean_vector(self, keys, weights=None, pre_normalize=True, post_normalize=False, ignore_missing=True):
        if len(keys) == 0:
            raise ValueError("cannot compute mean with no input")
        if isinstance(weights, list):
            weights = np.array(weights)
        if weights is None:
            weights = np.ones(len(keys))
        if len(keys) != weights.shape[0]:
            raise ValueError(
                "keys and weights array must have same number of elements"
            )

        mean = np.zeros(self.vector_size, self.vectors.dtype)

        total_weight = 0
        for idx, key in enumerate(keys):
            if isinstance(key, ndarray):
                mean += weights[idx] * key
                total_weight += abs(weights[idx])
            elif self.__contains__(key):
                vec = self.get_vector(key, norm=pre_normalize)
                mean += weights[idx] * vec
                total_weight += abs(weights[idx])
            elif not ignore_missing:
                raise KeyError(f"Key '{key}' not present in vocabulary")

        if(total_weight > 0):
            mean = mean / total_weight
        if post_normalize:
            mean = matutils.unitvec(mean).astype(REAL)
        return mean

    def add_vector(self, key, vector):
        target_index = self.next_index
        if target_index >= len(self) or self.index_to_key[target_index] is not None:
            target_index = len(self)
            warnings.warn(
                "Adding single vectors to a KeyedVectors which grows by one each time can be costly. "
                "Consider adding in batches or preallocating to the required size.",
                UserWarning)
            self.add_vectors([key], [vector])
            self.allocate_vecattrs()
            self.next_index = target_index + 1
        else:
            self.index_to_key[target_index] = key
            self.key_to_index[key] = target_index
            self.vectors[target_index] = vector
            self.next_index += 1
        return target_index

    def add_vectors(self, keys, weights, extras=None, replace=False):
        if isinstance(keys, _KEY_TYPES):
            keys = [keys]
            weights = np.array(weights).reshape(1, -1)
        elif isinstance(weights, list):
            weights = np.array(weights)
        if extras is None:
            extras = {}
        self.allocate_vecattrs(extras.keys(), [extras[k].dtype for k in extras.keys()])

        in_vocab_mask = np.zeros(len(keys), dtype=bool)
        for idx, key in enumerate(keys):
            if key in self:
                in_vocab_mask[idx] = True
        for idx in np.nonzero(~in_vocab_mask)[0]:
            key = keys[idx]
            self.key_to_index[key] = len(self.index_to_key)
            self.index_to_key.append(key)
        self.vectors = vstack((self.vectors, weights[~in_vocab_mask].astype(self.vectors.dtype)))
        for attr, extra in extras:
            self.expandos[attr] = np.vstack((self.expandos[attr], extra[~in_vocab_mask]))
        if replace:
            in_vocab_idxs = [self.get_index(keys[idx]) for idx in np.nonzero(in_vocab_mask)[0]]
            self.vectors[in_vocab_idxs] = weights[in_vocab_mask]
            for attr, extra in extras:
                self.expandos[attr][in_vocab_idxs] = extra[in_vocab_mask]

    def __setitem__(self, keys, weights):
        if not isinstance(keys, list):
            keys = [keys]
            weights = weights.reshape(1, -1)

        self.add_vectors(keys, weights, replace=True)

    def has_index_for(self, key):
        return self.get_index(key, -1) >= 0

    def __contains__(self, key):
        return self.has_index_for(key)

    def most_similar_to_given(self, key1, keys_list):
        return keys_list[argmax([self.similarity(key1, key) for key in keys_list])]

    def closer_than(self, key1, key2):
        all_distances = self.distances(key1)
        e1_index = self.get_index(key1)
        e2_index = self.get_index(key2)
        closer_node_indices = np.where(all_distances < all_distances[e2_index])[0]
        return [self.index_to_key[index] for index in closer_node_indices if index != e1_index]

    @deprecated("Use closer_than instead")
    def words_closer_than(self, word1, word2):
        return self.closer_than(word1, word2)

    def rank(self, key1, key2):
        return len(self.closer_than(key1, key2)) + 1

    @property
    def vectors_norm(self):
        raise AttributeError(
            "The `.vectors_norm` attribute is computed dynamically since Gensim 4.0.0. "
            "Use `.get_normed_vectors()` instead.\n"
            "See https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4"
        )

    @vectors_norm.setter
    def vectors_norm(self, _):
        pass

    def get_normed_vectors(self):
        self.fill_norms()
        return self.vectors / self.norms[..., np.newaxis]

    def fill_norms(self, force=False):
        if self.norms is None or force:
            self.norms = np.linalg.norm(self.vectors, axis=1)

    @property
    def index2entity(self):
        raise AttributeError(
            "The index2entity attribute has been replaced by index_to_key since Gensim 4.0.0.\n"
            "See https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4"
        )

    @index2entity.setter
    def index2entity(self, value):
        self.index_to_key = value

    @property
    def index2word(self):
        raise AttributeError(
            "The index2word attribute has been replaced by index_to_key since Gensim 4.0.0.\n"
            "See https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4"
        )

    @index2word.setter
    def index2word(self, value):
        self.index_to_key = value

    @property
    def vocab(self):
        raise AttributeError(
            "The vocab attribute was removed from KeyedVector in Gensim 4.0.0.\n"
            "Use KeyedVector's .key_to_index dict, .index_to_key list, and methods "
            ".get_vecattr(key, attr) and .set_vecattr(key, attr, new_val) instead.\n"
            "See https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4"
        )

    @vocab.setter
    def vocab(self, value):
        self.vocab()

    def sort_by_descending_frequency(self):
        if not len(self):
            return
        count_sorted_indexes = np.argsort(self.expandos['count'])[::-1]
        self.index_to_key = [self.index_to_key[idx] for idx in count_sorted_indexes]
        self.allocate_vecattrs()
        for k in self.expandos:
            self.expandos[k] = self.expandos[k][count_sorted_indexes]
        if len(self.vectors):
            logger.warning("sorting after vectors have been allocated is expensive & error-prone")
            self.vectors = self.vectors[count_sorted_indexes]
        self.key_to_index = {word: i for i, word in enumerate(self.index_to_key)}

    def save(self, *args, **kwargs):
        super(KeyedVectors, self).save(*args, **kwargs)

    def most_similar( self, positive=None, negative=None, topn=10, clip_start=0, clip_end=None, restrict_vocab=None,
                      indexer=None):
        if isinstance(topn, Integral) and topn < 1:
            return []
        positive = _ensure_list(positive)
        negative = _ensure_list(negative)
        self.fill_norms()
        clip_end = clip_end or len(self.vectors)
        if restrict_vocab:
            clip_start = 0
            clip_end = restrict_vocab
        keys = []
        weight = np.concatenate((np.ones(len(positive)), -1.0 * np.ones(len(negative))))
        for idx, item in enumerate(positive + negative):
            if isinstance(item, _EXTENDED_KEY_TYPES):
                keys.append(item)
            else:
                keys.append(item[0])
                weight[idx] = item[1]
        mean = self.get_mean_vector(keys, weight, pre_normalize=True, post_normalize=True, ignore_missing=False)
        all_keys = [
            self.get_index(key) for key in keys if isinstance(key, _KEY_TYPES) and self.has_index_for(key)
        ]
        if indexer is not None and isinstance(topn, int):
            return indexer.most_similar(mean, topn)
        dists = dot(self.vectors[clip_start:clip_end], mean) / self.norms[clip_start:clip_end]
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_keys), reverse=True)
        result = [
            (self.index_to_key[sim + clip_start], float(dists[sim]))
            for sim in best if (sim + clip_start) not in all_keys
        ]
        return result[:topn]

    def similar_by_word(self, word, topn=10, restrict_vocab=None):
        return self.similar_by_key(word, topn, restrict_vocab)

    def similar_by_key(self, key, topn=10, restrict_vocab=None):
        return self.most_similar(positive=[key], topn=topn, restrict_vocab=restrict_vocab)

    def similar_by_vector(self, vector, topn=10, restrict_vocab=None):
        return self.most_similar(positive=[vector], topn=topn, restrict_vocab=restrict_vocab)

    def wmdistance(self, document1, document2, norm=True):
        from pyemd import emd
        len_pre_oov1 = len(document1)
        len_pre_oov2 = len(document2)
        document1 = [token for token in document1 if token in self]
        document2 = [token for token in document2 if token in self]
        diff1 = len_pre_oov1 - len(document1)
        diff2 = len_pre_oov2 - len(document2)
        if diff1 > 0 or diff2 > 0:
            logger.info('Removed %d and %d OOV words from document 1 and 2 (respectively).', diff1, diff2)
        if not document1 or not document2:
            logger.warning("At least one of the documents had no words that were in the vocabulary.")
            return float('inf')
        dictionary = Dictionary(documents=[document1, document2])
        vocab_len = len(dictionary)
        if vocab_len == 1:
            return 0.0
        doclist1 = list(set(document1))
        doclist2 = list(set(document2))
        v1 = np.array([self.get_vector(token, norm=norm) for token in doclist1])
        v2 = np.array([self.get_vector(token, norm=norm) for token in doclist2])
        doc1_indices = dictionary.doc2idx(doclist1)
        doc2_indices = dictionary.doc2idx(doclist2)
        distance_matrix = zeros((vocab_len, vocab_len), dtype=double)
        distance_matrix[np.ix_(doc1_indices, doc2_indices)] = cdist(v1, v2)
        if abs(np_sum(distance_matrix)) < 1e-8:
            logger.info('The distance matrix is all zeros. Aborting (returning inf).')
            return float('inf')

        def nbow(document):
            d = zeros(vocab_len, dtype=double)
            nbow = dictionary.doc2bow(document)
            doc_len = len(document)
            for idx, freq in nbow:
                d[idx] = freq / float(doc_len)
            return d

        d1 = nbow(document1)
        d2 = nbow(document2)
        return emd(d1, d2, distance_matrix)

    def most_similar_cosmul(self, positive=None, negative=None, topn=10, restrict_vocab=None):
        if isinstance(topn, Integral) and topn < 1:
            return []
        positive = _ensure_list(positive)
        negative = _ensure_list(negative)
        self.init_sims()
        if isinstance(positive, str):
            positive = [positive]
        if isinstance(negative, str):
            negative = [negative]
        all_words = {
            self.get_index(word) for word in positive + negative
            if not isinstance(word, ndarray) and word in self.key_to_index
        }
        positive = [
            self.get_vector(word, norm=True) if isinstance(word, str) else word
            for word in positive
        ]
        negative = [
            self.get_vector(word, norm=True) if isinstance(word, str) else word
            for word in negative
        ]
        if not positive:
            raise ValueError("cannot compute similarity with no input")
        pos_dists = [((1 + dot(self.vectors, term) / self.norms) / 2) for term in positive]
        neg_dists = [((1 + dot(self.vectors, term) / self.norms) / 2) for term in negative]
        dists = prod(pos_dists, axis=0) / (prod(neg_dists, axis=0) + 0.000001)

        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
        result = [(self.index_to_key[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]

    def rank_by_centrality(self, words, use_norm=True):
        self.fill_norms()
        used_words = [word for word in words if word in self]
        if len(used_words) != len(words):
            ignored_words = set(words) - set(used_words)
            logger.warning("vectors for words %s are not present in the model, ignoring these words", ignored_words)
        if not used_words:
            raise ValueError("cannot select a word from an empty list")
        vectors = vstack([self.get_vector(word, norm=use_norm) for word in used_words]).astype(REAL)
        mean = self.get_mean_vector(vectors, post_normalize=True)
        dists = dot(vectors, mean)
        return sorted(zip(dists, used_words), reverse=True)

    def doesnt_match(self, words):
        return self.rank_by_centrality(words)[-1][1]

    @staticmethod
    def cosine_similarities(vector_1, vectors_all):
        norm = np.linalg.norm(vector_1)
        all_norms = np.linalg.norm(vectors_all, axis=1)
        dot_products = dot(vectors_all, vector_1)
        similarities = dot_products / (norm * all_norms)
        return similarities

    def distances(self, word_or_vector, other_words=()):
        if isinstance(word_or_vector, _KEY_TYPES):
            input_vector = self.get_vector(word_or_vector)
        else:
            input_vector = word_or_vector
        if not other_words:
            other_vectors = self.vectors
        else:
            other_indices = [self.get_index(word) for word in other_words]
            other_vectors = self.vectors[other_indices]
        return 1 - self.cosine_similarities(input_vector, other_vectors)

    def distance(self, w1, w2):
        return 1 - self.similarity(w1, w2)

    def similarity(self, w1, w2):
        return dot(matutils.unitvec(self[w1]), matutils.unitvec(self[w2]))

    def n_similarity(self, ws1, ws2):
        if not(len(ws1) and len(ws2)):
            raise ZeroDivisionError('At least one of the passed list is empty.')
        mean1 = self.get_mean_vector(ws1, pre_normalize=False)
        mean2 = self.get_mean_vector(ws2, pre_normalize=False)
        return dot(matutils.unitvec(mean1), matutils.unitvec(mean2))

    @staticmethod
    def _log_evaluate_word_analogies(section):
        correct, incorrect = len(section['correct']), len(section['incorrect'])
        if correct + incorrect == 0:
            return 0.0
        score = correct / (correct + incorrect)
        logger.info("%s: %.1f%% (%i/%i)", section['section'], 100.0 * score, correct, correct + incorrect)
        return score

    def evaluate_word_analogies(
            self, analogies, restrict_vocab=300000, case_insensitive=True,
            dummy4unknown=False, similarity_function='most_similar'):
        ok_keys = self.index_to_key[:restrict_vocab]
        if case_insensitive:
            ok_vocab = {k.upper(): self.get_index(k) for k in reversed(ok_keys)}
        else:
            ok_vocab = {k: self.get_index(k) for k in reversed(ok_keys)}
        oov = 0
        logger.info("Evaluating word analogies for top %i words in the model on %s", restrict_vocab, analogies)
        sections, section = [], None
        quadruplets_no = 0
        with utils.open(analogies, 'rb') as fin:
            for line_no, line in enumerate(fin):
                line = utils.to_unicode(line)
                if line.startswith(': '):
                    if section:
                        sections.append(section)
                        self._log_evaluate_word_analogies(section)
                    section = {'section': line.lstrip(': ').strip(), 'correct': [], 'incorrect': []}
                else:
                    if not section:
                        raise ValueError("Missing section header before line #%i in %s" % (line_no, analogies))
                    try:
                        if case_insensitive:
                            a, b, c, expected = [word.upper() for word in line.split()]
                        else:
                            a, b, c, expected = [word for word in line.split()]
                    except ValueError:
                        logger.info("Skipping invalid line #%i in %s", line_no, analogies)
                        continue
                    quadruplets_no += 1
                    if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
                        oov += 1
                        if dummy4unknown:
                            logger.debug('Zero accuracy for line #%d with OOV words: %s', line_no, line.strip())
                            section['incorrect'].append((a, b, c, expected))
                        else:
                            logger.debug("Skipping line #%i with OOV words: %s", line_no, line.strip())
                        continue
                    original_key_to_index = self.key_to_index
                    self.key_to_index = ok_vocab
                    ignore = {a, b, c}
                    predicted = None
                    sims = self.most_similar(positive=[b, c], negative=[a], topn=5, restrict_vocab=restrict_vocab)
                    self.key_to_index = original_key_to_index
                    for element in sims:
                        predicted = element[0].upper() if case_insensitive else element[0]
                        if predicted in ok_vocab and predicted not in ignore:
                            if predicted != expected:
                                logger.debug("%s: expected %s, predicted %s", line.strip(), expected, predicted)
                            break
                    if predicted == expected:
                        section['correct'].append((a, b, c, expected))
                    else:
                        section['incorrect'].append((a, b, c, expected))
        if section:
            sections.append(section)
            self._log_evaluate_word_analogies(section)

        total = {
            'section': 'Total accuracy',
            'correct': list(itertools.chain.from_iterable(s['correct'] for s in sections)),
            'incorrect': list(itertools.chain.from_iterable(s['incorrect'] for s in sections)),
        }

        oov_ratio = float(oov) / quadruplets_no * 100
        logger.info('Quadruplets with out-of-vocabulary words: %.1f%%', oov_ratio)
        if not dummy4unknown:
            logger.info(
                'NB: analogies containing OOV words were skipped from evaluation! '
                'To change this behavior, use "dummy4unknown=True"'
            )
        analogies_score = self._log_evaluate_word_analogies(total)
        sections.append(total)
        return analogies_score, sections

    @staticmethod
    def log_accuracy(section):
        correct, incorrect = len(section['correct']), len(section['incorrect'])
        if correct + incorrect > 0:
            logger.info(
                "%s: %.1f%% (%i/%i)",
                section['section'], 100.0 * correct / (correct + incorrect), correct, correct + incorrect,
            )

    @staticmethod
    def log_evaluate_word_pairs(pearson, spearman, oov, pairs):
        logger.info('Pearson correlation coefficient against %s: %.4f', pairs, pearson[0])
        logger.info('Spearman rank-order correlation coefficient against %s: %.4f', pairs, spearman[0])
        logger.info('Pairs with unknown words ratio: %.1f%%', oov)

    def evaluate_word_pairs(
            self, pairs, delimiter='\t', encoding='utf8',
            restrict_vocab=300000, case_insensitive=True, dummy4unknown=False,
        ):
        ok_keys = self.index_to_key[:restrict_vocab]
        if case_insensitive:
            ok_vocab = {k.upper(): self.get_index(k) for k in reversed(ok_keys)}
        else:
            ok_vocab = {k: self.get_index(k) for k in reversed(ok_keys)}

        similarity_gold = []
        similarity_model = []
        oov = 0

        original_key_to_index, self.key_to_index = self.key_to_index, ok_vocab
        try:
            with utils.open(pairs, encoding=encoding) as fin:
                for line_no, line in enumerate(fin):
                    if not line or line.startswith('#'):
                        continue
                    try:
                        if case_insensitive:
                            a, b, sim = [word.upper() for word in line.split(delimiter)]
                        else:
                            a, b, sim = [word for word in line.split(delimiter)]
                        sim = float(sim)
                    except (ValueError, TypeError):
                        logger.info('Skipping invalid line #%d in %s', line_no, pairs)
                        continue

                    if a not in ok_vocab or b not in ok_vocab:
                        oov += 1
                        if dummy4unknown:
                            logger.debug('Zero similarity for line #%d with OOV words: %s', line_no, line.strip())
                            similarity_model.append(0.0)
                            similarity_gold.append(sim)
                        else:
                            logger.info('Skipping line #%d with OOV words: %s', line_no, line.strip())
                        continue
                    similarity_gold.append(sim)
                    similarity_model.append(self.similarity(a, b))
        finally:
            self.key_to_index = original_key_to_index

        assert len(similarity_gold) == len(similarity_model)
        if not similarity_gold:
            raise ValueError(
                f"No valid similarity judgements found in {pairs}: either invalid format or "
                f"all are out-of-vocabulary in {self}"
            )
        spearman = stats.spearmanr(similarity_gold, similarity_model)
        pearson = stats.pearsonr(similarity_gold, similarity_model)
        if dummy4unknown:
            oov_ratio = float(oov) / len(similarity_gold) * 100
        else:
            oov_ratio = float(oov) / (len(similarity_gold) + oov) * 100

        logger.debug('Pearson correlation coefficient against %s: %f with p-value %f', pairs, pearson[0], pearson[1])
        logger.debug(
            'Spearman rank-order correlation coefficient against %s: %f with p-value %f',
            pairs, spearman[0], spearman[1]
        )
        logger.debug('Pairs with unknown words: %d', oov)
        self.log_evaluate_word_pairs(pearson, spearman, oov_ratio, pairs)
        return pearson, spearman, oov_ratio

    @deprecated(
        "Use fill_norms() instead. "
        "See https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4"
    )
    def init_sims(self, replace=False):
        self.fill_norms()
        if replace:
            logger.warning("destructive init_sims(replace=True) deprecated & no longer required for space-efficiency")
            self.unit_normalize_all()

    def unit_normalize_all(self):
        self.fill_norms()
        self.vectors /= self.norms[..., np.newaxis]
        self.norms = np.ones((len(self.vectors),))

    def relative_cosine_similarity(self, wa, wb, topn=10):

        sims = self.similar_by_word(wa, topn)
        if not sims:
            raise ValueError("Cannot calculate relative cosine similarity without any similar words.")
        rcs = float(self.similarity(wa, wb)) / (sum(sim for _, sim in sims))

        return rcs

    def save_word2vec_format(
            self, fname, fvocab=None, binary=False, total_vec=None, write_header=True,
            prefix='', append=False, sort_attr='count',
        ):
        if total_vec is None:
            total_vec = len(self.index_to_key)
        mode = 'wb' if not append else 'ab'

        if sort_attr in self.expandos:
            store_order_vocab_keys = sorted(self.key_to_index.keys(), key=lambda k: -self.get_vecattr(k, sort_attr))
        else:
            if fvocab is not None:
                raise ValueError(f"Cannot store vocabulary with '{sort_attr}' because that attribute does not exist")
            logger.warning(
                "attribute %s not present in %s; will store in internal index_to_key order",
                sort_attr, self,
            )
            store_order_vocab_keys = self.index_to_key

        if fvocab is not None:
            logger.info("storing vocabulary in %s", fvocab)
            with utils.open(fvocab, mode) as vout:
                for word in store_order_vocab_keys:
                    vout.write(f"{prefix}{word} {self.get_vecattr(word, sort_attr)}\n".encode('utf8'))

        logger.info("storing %sx%s projection weights into %s", total_vec, self.vector_size, fname)
        assert (len(self.index_to_key), self.vector_size) == self.vectors.shape
        index_id_count = 0
        for i, val in enumerate(self.index_to_key):
            if i != val:
                break
            index_id_count += 1
        keys_to_write = itertools.chain(range(0, index_id_count), store_order_vocab_keys)
        with utils.open(fname, mode) as fout:
            if write_header:
                fout.write(f"{total_vec} {self.vector_size}\n".encode('utf8'))
            for key in keys_to_write:
                key_vector = self[key]
                if binary:
                    fout.write(f"{prefix}{key} ".encode('utf8') + key_vector.astype(REAL).tobytes())
                else:
                    fout.write(f"{prefix}{key} {' '.join(repr(val) for val in key_vector)}\n".encode('utf8'))

    @classmethod
    def load_word2vec_format(
            cls, fname, fvocab=None, binary=False, encoding='utf8', unicode_errors='strict',
            limit=None, datatype=REAL, no_header=False,
        ):
        return _load_word2vec_format(
            cls, fname, fvocab=fvocab, binary=binary, encoding=encoding, unicode_errors=unicode_errors,
            limit=limit, datatype=datatype, no_header=no_header,
        )

    def intersect_word2vec_format(self, fname, lockf=0.0, binary=False, encoding='utf8', unicode_errors='strict'):
        overlap_count = 0
        logger.info("loading projection weights from %s", fname)
        with utils.open(fname, 'rb') as fin:
            header = utils.to_unicode(fin.readline(), encoding=encoding)
            vocab_size, vector_size = (int(x) for x in header.split())
            if not vector_size == self.vector_size:
                raise ValueError("incompatible vector size %d in file %s" % (vector_size, fname))
            if binary:
                binary_len = dtype(REAL).itemsize * vector_size
                for _ in range(vocab_size):
                    word = []
                    while True:
                        ch = fin.read(1)
                        if ch == b' ':
                            break
                        if ch != b'\n':
                            word.append(ch)
                    word = utils.to_unicode(b''.join(word), encoding=encoding, errors=unicode_errors)
                    weights = np.fromstring(fin.read(binary_len), dtype=REAL)
                    if word in self.key_to_index:
                        overlap_count += 1
                        self.vectors[self.get_index(word)] = weights
                        self.vectors_lockf[self.get_index(word)] = lockf
            else:
                for line_no, line in enumerate(fin):
                    parts = utils.to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")
                    if len(parts) != vector_size + 1:
                        raise ValueError("invalid vector on line %s (is this really the text format?)" % line_no)
                    word, weights = parts[0], [REAL(x) for x in parts[1:]]
                    if word in self.key_to_index:
                        overlap_count += 1
                        self.vectors[self.get_index(word)] = weights
                        self.vectors_lockf[self.get_index(word)] = lockf
        self.add_lifecycle_event(
            "intersect_word2vec_format",
            msg=f"merged {overlap_count} vectors into {self.vectors.shape} matrix from {fname}",
        )

    def vectors_for_all(self, keys: Iterable, allow_inference: bool = True,
                        copy_vecattrs: bool = False) -> 'KeyedVectors':
        vocab, seen = [], set()
        for key in keys:
            if key not in seen:
                seen.add(key)
                if key in (self if allow_inference else self.key_to_index):
                    vocab.append(key)
        kv = KeyedVectors(self.vector_size, len(vocab), dtype=self.vectors.dtype)
        for key in vocab:
            weights = self[key]
            _add_word_to_kv(kv, None, key, weights, len(vocab))
            if copy_vecattrs:
                for attr in self.expandos:
                    try:
                        kv.set_vecattr(key, attr, self.get_vecattr(key, attr))
                    except KeyError:
                        pass
        return kv

    def _upconvert_old_d2vkv(self):
        self.vocab = self.doctags
        self._upconvert_old_vocab()
        for k in self.key_to_index.keys():
            old_offset = self.get_vecattr(k, 'offset')
            true_index = old_offset + self.max_rawint + 1
            self.key_to_index[k] = true_index
        del self.expandos['offset']
        if self.max_rawint > -1:
            self.index_to_key = list(range(0, self.max_rawint + 1)) + self.offset2doctag
        else:
            self.index_to_key = self.offset2doctag
        self.vectors = self.vectors_docs
        del self.doctags
        del self.vectors_docs
        del self.count
        del self.max_rawint
        del self.offset2doctag

    def similarity_unseen_docs(self, *args, **kwargs):
        raise NotImplementedError("Call similarity_unseen_docs on a Doc2Vec model instead.")


Word2VecKeyedVectors = KeyedVectors
Doc2VecKeyedVectors = KeyedVectors
EuclideanKeyedVectors = KeyedVectors


class CompatVocab:
    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "%s<%s>" % (self.__class__.__name__, ', '.join(vals))


Vocab = CompatVocab


def _add_word_to_kv(kv, counts, word, weights, vocab_size):
    if kv.has_index_for(word):
        logger.warning("duplicate word '%s' in word2vec file, ignoring all but first", word)
        return
    word_id = kv.add_vector(word, weights)

    if counts is None:
        word_count = vocab_size - word_id
    elif word in counts:
        word_count = counts[word]
    else:
        logger.warning("vocabulary file is incomplete: '%s' is missing", word)
        word_count = None
    kv.set_vecattr(word, 'count', word_count)


def _add_bytes_to_kv(kv, counts, chunk, vocab_size, vector_size, datatype, unicode_errors, encoding):
    start = 0
    processed_words = 0
    bytes_per_vector = vector_size * dtype(REAL).itemsize
    max_words = vocab_size - kv.next_index
    assert max_words > 0
    for _ in range(max_words):
        i_space = chunk.find(b' ', start)
        i_vector = i_space + 1
        if i_space == -1 or (len(chunk) - i_vector) < bytes_per_vector:
            break
        word = chunk[start:i_space].decode(encoding, errors=unicode_errors)
        word = word.lstrip('\n')
        vector = frombuffer(chunk, offset=i_vector, count=vector_size, dtype=REAL).astype(datatype)
        _add_word_to_kv(kv, counts, word, vector, vocab_size)
        start = i_vector + bytes_per_vector
        processed_words += 1
    return processed_words, chunk[start:]


def _word2vec_read_binary(
        fin, kv, counts, vocab_size, vector_size, datatype, unicode_errors, binary_chunk_size,
        encoding="utf-8",
    ):
    chunk = b''
    tot_processed_words = 0

    while tot_processed_words < vocab_size:
        new_chunk = fin.read(binary_chunk_size)
        chunk += new_chunk
        processed_words, chunk = _add_bytes_to_kv(
            kv, counts, chunk, vocab_size, vector_size, datatype, unicode_errors, encoding)
        tot_processed_words += processed_words
        if len(new_chunk) < binary_chunk_size:
            break
    if tot_processed_words != vocab_size:
        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")


def _word2vec_read_text(fin, kv, counts, vocab_size, vector_size, datatype, unicode_errors, encoding):
    for line_no in range(vocab_size):
        line = fin.readline()
        if line == b'':
            raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
        word, weights = _word2vec_line_to_vector(line, datatype, unicode_errors, encoding)
        _add_word_to_kv(kv, counts, word, weights, vocab_size)


def _word2vec_line_to_vector(line, datatype, unicode_errors, encoding):
    parts = utils.to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")
    word, weights = parts[0], [datatype(x) for x in parts[1:]]
    return word, weights


def _word2vec_detect_sizes_text(fin, limit, datatype, unicode_errors, encoding):
    vector_size = None
    for vocab_size in itertools.count():
        line = fin.readline()
        if line == b'' or vocab_size == limit:
            break
        if vector_size:
            continue
        word, weights = _word2vec_line_to_vector(line, datatype, unicode_errors, encoding)
        vector_size = len(weights)
    return vocab_size, vector_size


def _load_word2vec_format(
        cls, fname, fvocab=None, binary=False, encoding='utf8', unicode_errors='strict',
        limit=sys.maxsize, datatype=REAL, no_header=False, binary_chunk_size=100 * 1024,
    ):
    counts = None
    if fvocab is not None:
        logger.info("loading word counts from %s", fvocab)
        counts = {}
        with utils.open(fvocab, 'rb') as fin:
            for line in fin:
                word, count = utils.to_unicode(line, errors=unicode_errors).strip().split()
                counts[word] = int(count)
    logger.info("loading projection weights from %s", fname)
    with utils.open(fname, 'rb') as fin:
        if no_header:
            if binary:
                raise NotImplementedError("no_header only available for text-format files")
            else:
                vocab_size, vector_size = _word2vec_detect_sizes_text(fin, limit, datatype, unicode_errors, encoding)
            fin.close()
            fin = utils.open(fname, 'rb')
        else:
            header = utils.to_unicode(fin.readline(), encoding=encoding)
            vocab_size, vector_size = [int(x) for x in header.split()]
        if limit:
            vocab_size = min(vocab_size, limit)
        kv = cls(vector_size, vocab_size, dtype=datatype)

        if binary:
            _word2vec_read_binary(
                fin, kv, counts, vocab_size, vector_size, datatype, unicode_errors, binary_chunk_size, encoding
            )
        else:
            _word2vec_read_text(fin, kv, counts, vocab_size, vector_size, datatype, unicode_errors, encoding)
    if kv.vectors.shape[0] != len(kv):
        logger.info(
            "duplicate words detected, shrinking matrix size from %i to %i",
            kv.vectors.shape[0], len(kv),
        )
        kv.vectors = ascontiguousarray(kv.vectors[: len(kv)])
    assert (len(kv), vector_size) == kv.vectors.shape
    kv.add_lifecycle_event(
        "load_word2vec_format",
        msg=f"loaded {kv.vectors.shape} matrix of type {kv.vectors.dtype} from {fname}",
        binary=binary, encoding=encoding,
    )
    return kv


def load_word2vec_format(*args, **kwargs):
    return KeyedVectors.load_word2vec_format(*args, **kwargs)


def pseudorandom_weak_vector(size, seed_string=None, hashfxn=hash):
    if seed_string:
        once = np.random.Generator(np.random.SFC64(hashfxn(seed_string) & 0xffffffff))
    else:
        once = utils.default_prng
    return (once.random(size).astype(REAL) - 0.5) / size


def prep_vectors(target_shape, prior_vectors=None, seed=0, dtype=REAL):
    if prior_vectors is None:
        prior_vectors = np.zeros((0, 0))
    if prior_vectors.shape == target_shape:
        return prior_vectors
    target_count, vector_size = target_shape
    rng = np.random.default_rng(seed=seed)
    new_vectors = rng.random(target_shape, dtype=dtype)
    new_vectors *= 2.0
    new_vectors -= 1.0
    new_vectors /= vector_size
    new_vectors[0:prior_vectors.shape[0], 0:prior_vectors.shape[1]] = prior_vectors
    return new_vectors
