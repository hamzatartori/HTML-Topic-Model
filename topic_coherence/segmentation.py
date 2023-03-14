import logging

logger = logging.getLogger(__name__)


def s_one_pre(topics):

    s_one_pre_res = []

    for top_words in topics:
        s_one_pre_t = []
        for w_prime_index, w_prime in enumerate(top_words[1:]):
            for w_star in top_words[:w_prime_index + 1]:
                s_one_pre_t.append((w_prime, w_star))
        s_one_pre_res.append(s_one_pre_t)

    return s_one_pre_res


def s_one_one(topics):

    s_one_one_res = []

    for top_words in topics:
        s_one_one_t = []
        for w_prime_index, w_prime in enumerate(top_words):
            for w_star_index, w_star in enumerate(top_words):
                if w_prime_index == w_star_index:
                    continue
                else:
                    s_one_one_t.append((w_prime, w_star))
        s_one_one_res.append(s_one_one_t)

    return s_one_one_res


def s_one_set(topics):

    s_one_set_res = []

    for top_words in topics:
        s_one_set_t = []
        for w_prime in top_words:
            s_one_set_t.append((w_prime, top_words))
        s_one_set_res.append(s_one_set_t)

    return s_one_set_res
