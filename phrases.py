import logging
import itertools
from math import log
import pickle
from inspect import getfullargspec as getargspec
import time
import utils, interfaces


logger = logging.getLogger(__name__)

NEGATIVE_INFINITY = float('-inf')

ENGLISH_CONNECTOR_WORDS = frozenset(
    " a an the " 
    " for of with without at from to in on by "  
    " and or "
    .split()
)


def original_scorer(worda_count, wordb_count, bigram_count, len_vocab, min_count, corpus_word_count):
    denom = worda_count * wordb_count
    if denom == 0:
        return NEGATIVE_INFINITY
    return (bigram_count - min_count) / float(denom) * len_vocab


def npmi_scorer(worda_count, wordb_count, bigram_count, len_vocab, min_count, corpus_word_count):
    if bigram_count >= min_count:
        corpus_word_count = float(corpus_word_count)
        pa = worda_count / corpus_word_count
        pb = wordb_count / corpus_word_count
        pab = bigram_count / corpus_word_count
        try:
            return log(pab / (pa * pb)) / -log(pab)
        except ValueError:
            return NEGATIVE_INFINITY
    else:
        return NEGATIVE_INFINITY


def _is_single(obj):
    obj_iter = iter(obj)
    temp_iter = obj_iter
    try:
        peek = next(obj_iter)
        obj_iter = itertools.chain([peek], obj_iter)
    except StopIteration:
        return True, obj
    if isinstance(peek, str):
        return True, obj_iter
    if temp_iter is obj:
        return False, obj_iter
    return False, obj


class _PhrasesTransformation(interfaces.TransformationABC):
    def __init__(self, connector_words):
        self.connector_words = frozenset(connector_words)

    def score_candidate(self, word_a, word_b, in_between):
        raise NotImplementedError("ABC: override this method in child classes")

    def analyze_sentence(self, sentence):
        start_token, in_between = None, []
        for word in sentence:
            if word not in self.connector_words:
                if start_token:
                    phrase, score = self.score_candidate(start_token, word, in_between)
                    if score is not None:
                        yield phrase, score
                        start_token, in_between = None, []
                    else:
                        yield start_token, None
                        for w in in_between:
                            yield w, None
                        start_token, in_between = word, []
                else:
                    start_token, in_between = word, []
                if start_token:
                    in_between.append(word)
                else:
                    yield word, None
        if start_token:
            yield start_token, None
            for w in in_between:
                yield w, None

    def __getitem__(self, sentence):
        is_single, sentence = _is_single(sentence)
        if not is_single:
            return self._apply(sentence)
        return [token for token, _ in self.analyze_sentence(sentence)]

    def find_phrases(self, sentences):
        result = {}
        for sentence in sentences:
            for phrase, score in self.analyze_sentence(sentence):
                if score is not None:
                    result[phrase] = score
        return result

    @classmethod
    def load(cls, *args, **kwargs):
        model = super(_PhrasesTransformation, cls).load(*args, **kwargs)
        try:
            phrasegrams = getattr(model, "phrasegrams", {})
            component, score = next(iter(phrasegrams.items()))
            if isinstance(score, tuple):
                model.phrasegrams = {
                    str(model.delimiter.join(key), encoding='utf8'): val[1]
                    for key, val in phrasegrams.items()
                }
            elif isinstance(component, tuple):
                model.phrasegrams = {
                    str(model.delimiter.join(key), encoding='utf8'): val
                    for key, val in phrasegrams.items()
                }
        except StopIteration:
            pass
        if not hasattr(model, 'scoring'):
            logger.warning('older version of %s loaded without scoring function', cls.__name__)
            logger.warning('setting pluggable scoring method to original_scorer for compatibility')
            model.scoring = original_scorer
        if hasattr(model, 'scoring'):
            if isinstance(model.scoring, str):
                if model.scoring == 'default':
                    logger.warning('older version of %s loaded with "default" scoring parameter', cls.__name__)
                    logger.warning('setting scoring method to original_scorer for compatibility')
                    model.scoring = original_scorer
                elif model.scoring == 'npmi':
                    logger.warning('older version of %s loaded with "npmi" scoring parameter', cls.__name__)
                    logger.warning('setting scoring method to npmi_scorer for compatibility')
                    model.scoring = npmi_scorer
                else:
                    raise ValueError(f'failed to load {cls.__name__} model, unknown scoring "{model.scoring}"')
        if not hasattr(model, "connector_words"):
            if hasattr(model, "common_terms"):
                model.connector_words = model.common_terms
                del model.common_terms
            else:
                logger.warning('loaded older version of %s, setting connector_words to an empty set', cls.__name__)
                model.connector_words = frozenset()
        if not hasattr(model, 'corpus_word_count'):
            logger.warning('older version of %s loaded without corpus_word_count', cls.__name__)
            logger.warning('setting corpus_word_count to 0, do not use it in your scoring function')
            model.corpus_word_count = 0

        if getattr(model, 'vocab', None):
            word = next(iter(model.vocab))
            if not isinstance(word, str):
                logger.info("old version of %s loaded, upgrading %i words in memory", cls.__name__, len(model.vocab))
                logger.info("re-save the loaded model to avoid this upgrade in the future")
                vocab = {}
                for key, value in model.vocab.items():
                    vocab[str(key, encoding='utf8')] = value
                model.vocab = vocab
        if not isinstance(model.delimiter, str):
            model.delimiter = str(model.delimiter, encoding='utf8')
        return model


class Phrases(_PhrasesTransformation):
    def __init__( self, sentences=None, min_count=5, threshold=10.0, max_vocab_size=40000000, delimiter='_',
                  progress_per=10000, scoring='default', connector_words=frozenset()):
        super().__init__(connector_words=connector_words)
        if min_count <= 0:
            raise ValueError("min_count should be at least 1")
        if threshold <= 0 and scoring == 'default':
            raise ValueError("threshold should be positive for default scoring")
        if scoring == 'npmi' and (threshold < -1 or threshold > 1):
            raise ValueError("threshold should be between -1 and 1 for npmi scoring")
        if isinstance(scoring, str):
            if scoring == 'default':
                scoring = original_scorer
            elif scoring == 'npmi':
                scoring = npmi_scorer
            else:
                raise ValueError(f'unknown scoring method string {scoring} specified')

        scoring_params = [
            'worda_count', 'wordb_count', 'bigram_count', 'len_vocab', 'min_count', 'corpus_word_count',
        ]
        if callable(scoring):
            missing = [param for param in scoring_params if param not in getargspec(scoring)[0]]
            if not missing:
                self.scoring = scoring
            else:
                raise ValueError(f'scoring function missing expected parameters {missing}')
        self.min_count = min_count
        self.threshold = threshold
        self.max_vocab_size = max_vocab_size
        self.vocab = {}
        self.min_reduce = 1
        self.delimiter = delimiter
        self.progress_per = progress_per
        self.corpus_word_count = 0
        try:
            pickle.loads(pickle.dumps(self.scoring))
        except pickle.PickleError:
            raise pickle.PickleError(f'Custom scoring function in {self.__class__.__name__} must be pickle-able')

        if sentences is not None:
            start = time.time()
            self.add_vocab(sentences)
            self.add_lifecycle_event("created", msg=f"built {self} in {time.time() - start:.2f}s")

    def __str__(self):
        return "%s<%i vocab, min_count=%s, threshold=%s, max_vocab_size=%s>" % (
            self.__class__.__name__, len(self.vocab), self.min_count,
            self.threshold, self.max_vocab_size,
        )

    @staticmethod
    def _learn_vocab(sentences, max_vocab_size, delimiter, connector_words, progress_per):
        sentence_no, total_words, min_reduce = -1, 0, 1
        vocab = {}
        logger.info("collecting all words and their counts")
        for sentence_no, sentence in enumerate(sentences):
            if sentence_no % progress_per == 0:
                logger.info(
                    "PROGRESS: at sentence #%i, processed %i words and %i word types",
                    sentence_no, total_words, len(vocab),
                )
            start_token, in_between = None, []
            for word in sentence:
                if word not in connector_words:
                    vocab[word] = vocab.get(word, 0) + 1
                    if start_token is not None:
                        phrase_tokens = itertools.chain([start_token], in_between, [word])
                        joined_phrase_token = delimiter.join(phrase_tokens)
                        vocab[joined_phrase_token] = vocab.get(joined_phrase_token, 0) + 1
                    start_token, in_between = word, []
                elif start_token is not None:
                    in_between.append(word)
                total_words += 1
            if len(vocab) > max_vocab_size:
                utils.prune_vocab(vocab, min_reduce)
                min_reduce += 1
        logger.info(
            "collected %i token types (unigram + bigrams) from a corpus of %i words and %i sentences",
            len(vocab), total_words, sentence_no + 1,
        )
        return min_reduce, vocab, total_words

    def add_vocab(self, sentences):
        min_reduce, vocab, total_words = self._learn_vocab(
            sentences, max_vocab_size=self.max_vocab_size, delimiter=self.delimiter,
            progress_per=self.progress_per, connector_words=self.connector_words,
        )

        self.corpus_word_count += total_words
        if self.vocab:
            logger.info("merging %i counts into %s", len(vocab), self)
            self.min_reduce = max(self.min_reduce, min_reduce)
            for word, count in vocab.items():
                self.vocab[word] = self.vocab.get(word, 0) + count
            if len(self.vocab) > self.max_vocab_size:
                utils.prune_vocab(self.vocab, self.min_reduce)
                self.min_reduce += 1
        else:
            self.vocab = vocab
        logger.info("merged %s", self)

    def score_candidate(self, word_a, word_b, in_between):
        word_a_cnt = self.vocab.get(word_a, 0)
        if word_a_cnt <= 0:
            return None, None
        word_b_cnt = self.vocab.get(word_b, 0)
        if word_b_cnt <= 0:
            return None, None
        phrase = self.delimiter.join([word_a] + in_between + [word_b])
        phrase_cnt = self.vocab.get(phrase, 0)
        if phrase_cnt <= 0:
            return None, None
        score = self.scoring(
            worda_count=word_a_cnt, wordb_count=word_b_cnt, bigram_count=phrase_cnt,
            len_vocab=len(self.vocab), min_count=self.min_count, corpus_word_count=self.corpus_word_count,
        )
        if score <= self.threshold:
            return None, None
        return phrase, score

    def freeze(self):
        return FrozenPhrases(self)

    def export_phrases(self):
        result, source_vocab = {}, self.vocab
        for token in source_vocab:
            unigrams = token.split(self.delimiter)
            if len(unigrams) < 2:
                continue
            phrase, score = self.score_candidate(unigrams[0], unigrams[-1], unigrams[1:-1])
            if score is not None:
                result[phrase] = score
        return result


class FrozenPhrases(_PhrasesTransformation):
    def __init__(self, phrases_model):
        self.threshold = phrases_model.threshold
        self.min_count = phrases_model.min_count
        self.delimiter = phrases_model.delimiter
        self.scoring = phrases_model.scoring
        self.connector_words = phrases_model.connector_words
        logger.info('exporting phrases from %s', phrases_model)
        start = time.time()
        self.phrasegrams = phrases_model.export_phrases()
        self.add_lifecycle_event("created", msg=f"exported {self} from {phrases_model} in {time.time() - start:.2f}s")

    def __str__(self):
        return "%s<%i phrases, min_count=%s, threshold=%s>" % (self.__class__.__name__, len(self.phrasegrams),
                                                               self.min_count, self.threshold)

    def score_candidate(self, word_a, word_b, in_between):
        phrase = self.delimiter.join([word_a] + in_between + [word_b])
        score = self.phrasegrams.get(phrase, NEGATIVE_INFINITY)
        if score > self.threshold:
            return phrase, score
        return None, None


Phraser = FrozenPhrases
