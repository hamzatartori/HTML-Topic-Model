import itertools
import logging

from topic_coherence.text_analysis import (
    CorpusAccumulator, WordOccurrenceAccumulator, ParallelWordOccurrenceAccumulator,
    WordVectorsAccumulator,
)

logger = logging.getLogger(__name__)


def p_boolean_document(corpus, segmented_topics):

    top_ids = unique_ids_from_segments(segmented_topics)
    return CorpusAccumulator(top_ids).accumulate(corpus)


def p_boolean_sliding_window(texts, segmented_topics, dictionary, window_size, processes=1):

    top_ids = unique_ids_from_segments(segmented_topics)
    if processes <= 1:
        accumulator = WordOccurrenceAccumulator(top_ids, dictionary)
    else:
        accumulator = ParallelWordOccurrenceAccumulator(processes, top_ids, dictionary)
    logger.info("using %s to estimate probabilities from sliding windows", accumulator)
    return accumulator.accumulate(texts, window_size)


def p_word2vec(texts, segmented_topics, dictionary, window_size=None, processes=1, model=None):

    top_ids = unique_ids_from_segments(segmented_topics)
    accumulator = WordVectorsAccumulator(
        top_ids, dictionary, model, window=window_size, workers=processes)
    return accumulator.accumulate(texts, window_size)


def unique_ids_from_segments(segmented_topics):
    unique_ids = set()
    for s_i in segmented_topics:
        for word_id in itertools.chain.from_iterable(s_i):
            if hasattr(word_id, '__iter__'):
                unique_ids.update(word_id)
            else:
                unique_ids.add(word_id)

    return unique_ids
