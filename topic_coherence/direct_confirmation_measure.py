import logging
import numpy as np

logger = logging.getLogger(__name__)

EPSILON = 1e-12


def log_conditional_probability(segmented_topics, accumulator, with_std=False, with_support=False):
    topic_coherences = []
    num_docs = float(accumulator.num_docs)
    for s_i in segmented_topics:
        segment_sims = []
        for w_prime, w_star in s_i:
            try:
                w_star_count = accumulator[w_star]
                co_occur_count = accumulator[w_prime, w_star]
                m_lc_i = np.log(((co_occur_count / num_docs) + EPSILON) / (w_star_count / num_docs))
            except KeyError:
                m_lc_i = 0.0
            except ZeroDivisionError:
                m_lc_i = 0.0
            segment_sims.append(m_lc_i)
        topic_coherences.append(aggregate_segment_sims(segment_sims, with_std, with_support))
    return topic_coherences


def aggregate_segment_sims(segment_sims, with_std, with_support):
    mean = np.mean(segment_sims)
    stats = [mean]
    if with_std:
        stats.append(np.std(segment_sims))
    if with_support:
        stats.append(len(segment_sims))
    return stats[0] if len(stats) == 1 else tuple(stats)


def log_ratio_measure(segmented_topics, accumulator, normalize=False, with_std=False, with_support=False):
    topic_coherences = []
    num_docs = float(accumulator.num_docs)
    for s_i in segmented_topics:
        segment_sims = []
        for w_prime, w_star in s_i:
            w_prime_count = accumulator[w_prime]
            w_star_count = accumulator[w_star]
            co_occur_count = accumulator[w_prime, w_star]
            if normalize:
                numerator = log_ratio_measure([[(w_prime, w_star)]], accumulator)[0]
                co_doc_prob = co_occur_count / num_docs
                m_lr_i = numerator / (-np.log(co_doc_prob + EPSILON))
            else:
                numerator = (co_occur_count / num_docs) + EPSILON
                denominator = (w_prime_count / num_docs) * (w_star_count / num_docs)
                m_lr_i = np.log(numerator / denominator)
            segment_sims.append(m_lr_i)
        topic_coherences.append(aggregate_segment_sims(segment_sims, with_std, with_support))
    return topic_coherences
