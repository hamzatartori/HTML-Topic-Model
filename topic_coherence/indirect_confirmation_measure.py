import itertools
import logging
import numpy as np
import scipy.sparse as sps

from topic_coherence.direct_confirmation_measure import aggregate_segment_sims, log_ratio_measure

logger = logging.getLogger(__name__)


def word2vec_similarity(segmented_topics, accumulator, with_std=False, with_support=False):
    topic_coherences = []
    total_oov = 0
    for topic_index, topic_segments in enumerate(segmented_topics):
        segment_sims = []
        num_oov = 0
        for w_prime, w_star in topic_segments:
            if not hasattr(w_prime, '__iter__'):
                w_prime = [w_prime]
            if not hasattr(w_star, '__iter__'):
                w_star = [w_star]
            try:
                segment_sims.append(accumulator.ids_similarity(w_prime, w_star))
            except ZeroDivisionError:
                num_oov += 1
        if num_oov > 0:
            total_oov += 1
            logger.warning(
                "%d terms for topic %d are not in word2vec model vocabulary",
                num_oov, topic_index)
        topic_coherences.append(aggregate_segment_sims(segment_sims, with_std, with_support))
    if total_oov > 0:
        logger.warning("%d terms for are not in word2vec model vocabulary", total_oov)
    return topic_coherences


def cosine_similarity(segmented_topics, accumulator, topics, measure='nlr',
                      gamma=1, with_std=False, with_support=False):
    context_vectors = ContextVectorComputer(measure, topics, accumulator, gamma)
    topic_coherences = []
    for topic_words, topic_segments in zip(topics, segmented_topics):
        topic_words = tuple(topic_words)
        segment_sims = np.zeros(len(topic_segments))
        for i, (w_prime, w_star) in enumerate(topic_segments):
            w_prime_cv = context_vectors[w_prime, topic_words]
            w_star_cv = context_vectors[w_star, topic_words]
            segment_sims[i] = _cossim(w_prime_cv, w_star_cv)
        topic_coherences.append(aggregate_segment_sims(segment_sims, with_std, with_support))
    return topic_coherences


class ContextVectorComputer:
    def __init__(self, measure, topics, accumulator, gamma):
        if measure == 'nlr':
            self.similarity = _pair_npmi
        else:
            raise ValueError(
                "The direct confirmation measure you entered is not currently supported.")
        self.mapping = _map_to_contiguous(topics)
        self.vocab_size = len(self.mapping)
        self.accumulator = accumulator
        self.gamma = gamma
        self.sim_cache = {}
        self.context_vector_cache = {}

    def __getitem__(self, idx):
        return self.compute_context_vector(*idx)

    def compute_context_vector(self, segment_word_ids, topic_word_ids):
        key = _key_for_segment(segment_word_ids, topic_word_ids)
        context_vector = self.context_vector_cache.get(key, None)
        if context_vector is None:
            context_vector = self._make_seg(segment_word_ids, topic_word_ids)
            self.context_vector_cache[key] = context_vector
        return context_vector

    def _make_seg(self, segment_word_ids, topic_word_ids):
        context_vector = sps.lil_matrix((self.vocab_size, 1))
        if not hasattr(segment_word_ids, '__iter__'):
            segment_word_ids = (segment_word_ids,)
        for w_j in topic_word_ids:
            idx = (self.mapping[w_j], 0)
            for pair in (tuple(sorted((w_i, w_j))) for w_i in segment_word_ids):
                if pair not in self.sim_cache:
                    self.sim_cache[pair] = self.similarity(pair, self.accumulator)
                context_vector[idx] += self.sim_cache[pair] ** self.gamma
        return context_vector.tocsr()


def _pair_npmi(pair, accumulator):
    return log_ratio_measure([[pair]], accumulator, True)[0]


def _cossim(cv1, cv2):
    return cv1.T.dot(cv2)[0, 0] / (_magnitude(cv1) * _magnitude(cv2))


def _magnitude(sparse_vec):
    return np.sqrt(np.sum(sparse_vec.data ** 2))


def _map_to_contiguous(ids_iterable):
    uniq_ids = {}
    n = 0
    for id_ in itertools.chain.from_iterable(ids_iterable):
        if id_ not in uniq_ids:
            uniq_ids[id_] = n
            n += 1
    return uniq_ids


def _key_for_segment(segment, topic_words):
    segment_key = tuple(segment) if hasattr(segment, '__iter__') else segment
    return segment_key, topic_words
