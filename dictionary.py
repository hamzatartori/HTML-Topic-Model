from collections import defaultdict
from collections.abc import Mapping
import logging
import itertools
from typing import Optional, List, Tuple

import utils


logger = logging.getLogger(__name__)


class Dictionary(utils.SaveLoad, Mapping):
    def __init__(self, documents=None, prune_at=2000000):
        self.token2id = {}
        self.id2token = {}
        self.cfs = {}
        self.dfs = {}
        self.num_docs = 0
        self.num_pos = 0
        self.num_nnz = 0
        if documents is not None:
            self.add_documents(documents, prune_at=prune_at)
            self.add_lifecycle_event(
                "created",
                msg=f"built {self} from {self.num_docs} documents (total {self.num_pos} corpus positions)",
            )

    def __getitem__(self, tokenid):
        if len(self.id2token) != len(self.token2id):
            self.id2token = utils.revdict(self.token2id)
        return self.id2token[tokenid]

    def __iter__(self):
        return iter(self.keys())

    iterkeys = __iter__

    def iteritems(self):
        return self.items()

    def itervalues(self):
        return self.values()

    def keys(self):
        return list(self.token2id.values())

    def __len__(self):
        return len(self.token2id)

    def __str__(self):
        some_keys = list(itertools.islice(self.token2id.keys(), 5))
        return "%s<%i unique tokens: %s%s>" % (
            self.__class__.__name__, len(self), some_keys, '...' if len(self) > 5 else ''
        )

    @staticmethod
    def from_documents(documents):
        return Dictionary(documents=documents)

    def add_documents(self, documents, prune_at=2000000):
        for docno, document in enumerate(documents):
            if docno % 10000 == 0:
                if prune_at is not None and len(self) > prune_at:
                    self.filter_extremes(no_below=0, no_above=1.0, keep_n=prune_at)
                logger.info("adding document #%i to %s", docno, self)
            self.doc2bow(document, allow_update=True)

        logger.info("built %s from %i documents (total %i corpus positions)", self, self.num_docs, self.num_pos)

    def doc2bow(self, document, allow_update=False, return_missing=False):
        if isinstance(document, str):
            raise TypeError("doc2bow expects an array of unicode tokens on input, not a single string")
        counter = defaultdict(int)
        for w in document:
            counter[w if isinstance(w, str) else str(w, 'utf-8')] += 1

        token2id = self.token2id
        if allow_update or return_missing:
            missing = sorted(x for x in counter.items() if x[0] not in token2id)
            if allow_update:
                for w, _ in missing:
                    token2id[w] = len(token2id)
        result = {token2id[w]: freq for w, freq in counter.items() if w in token2id}

        if allow_update:
            self.num_docs += 1
            self.num_pos += sum(counter.values())
            self.num_nnz += len(result)
            for tokenid, freq in result.items():
                self.cfs[tokenid] = self.cfs.get(tokenid, 0) + freq
                self.dfs[tokenid] = self.dfs.get(tokenid, 0) + 1
        result = sorted(result.items())
        if return_missing:
            return result, dict(missing)
        else:
            return result

    def doc2idx(self, document, unknown_word_index=-1):
        if isinstance(document, str):
            raise TypeError("doc2idx expects an array of unicode tokens on input, not a single string")

        document = [word if isinstance(word, str) else str(word, 'utf-8') for word in document]
        return [self.token2id.get(word, unknown_word_index) for word in document]

    def filter_extremes(self, no_below=5, no_above=0.5, keep_n=100000, keep_tokens=None):
        no_above_abs = int(no_above * self.num_docs)
        if keep_tokens:
            keep_ids = {self.token2id[v] for v in keep_tokens if v in self.token2id}
            good_ids = [
                v for v in self.token2id.values()
                if no_below <= self.dfs.get(v, 0) <= no_above_abs or v in keep_ids
            ]
            good_ids.sort(key=lambda x: self.num_docs if x in keep_ids else self.dfs.get(x, 0), reverse=True)
        else:
            good_ids = [
                v for v in self.token2id.values()
                if no_below <= self.dfs.get(v, 0) <= no_above_abs
            ]
            good_ids.sort(key=self.dfs.get, reverse=True)
        if keep_n is not None:
            good_ids = good_ids[:keep_n]
        bad_words = [(self[idx], self.dfs.get(idx, 0)) for idx in set(self).difference(good_ids)]
        logger.info("discarding %i tokens: %s...", len(self) - len(good_ids), bad_words[:10])
        logger.info(
            "keeping %i tokens which were in no less than %i and no more than %i (=%.1f%%) documents",
            len(good_ids), no_below, no_above_abs, 100.0 * no_above
        )

        self.filter_tokens(good_ids=good_ids)
        logger.info("resulting dictionary: %s", self)

    def filter_n_most_frequent(self, remove_n):
        most_frequent_ids = (v for v in self.token2id.values())
        most_frequent_ids = sorted(most_frequent_ids, key=self.dfs.get, reverse=True)
        most_frequent_ids = most_frequent_ids[:remove_n]
        most_frequent_words = [(self[idx], self.dfs.get(idx, 0)) for idx in most_frequent_ids]
        logger.info("discarding %i tokens: %s...", len(most_frequent_ids), most_frequent_words[:10])

        self.filter_tokens(bad_ids=most_frequent_ids)
        logger.info("resulting dictionary: %s", self)

    def filter_tokens(self, bad_ids=None, good_ids=None):
        if bad_ids is not None:
            bad_ids = set(bad_ids)
            self.token2id = {token: tokenid for token, tokenid in self.token2id.items() if tokenid not in bad_ids}
            self.cfs = {tokenid: freq for tokenid, freq in self.cfs.items() if tokenid not in bad_ids}
            self.dfs = {tokenid: freq for tokenid, freq in self.dfs.items() if tokenid not in bad_ids}
        if good_ids is not None:
            good_ids = set(good_ids)
            self.token2id = {token: tokenid for token, tokenid in self.token2id.items() if tokenid in good_ids}
            self.cfs = {tokenid: freq for tokenid, freq in self.cfs.items() if tokenid in good_ids}
            self.dfs = {tokenid: freq for tokenid, freq in self.dfs.items() if tokenid in good_ids}
        self.compactify()

    def compactify(self):
        logger.debug("rebuilding dictionary, shrinking gaps")
        idmap = dict(zip(sorted(self.token2id.values()), range(len(self.token2id))))
        self.token2id = {token: idmap[tokenid] for token, tokenid in self.token2id.items()}
        self.id2token = {}
        self.dfs = {idmap[tokenid]: freq for tokenid, freq in self.dfs.items()}
        self.cfs = {idmap[tokenid]: freq for tokenid, freq in self.cfs.items()}

    def save_as_text(self, fname, sort_by_word=True):
        logger.info("saving dictionary mapping to %s", fname)
        with utils.open(fname, 'wb') as fout:
            numdocs_line = "%d\n" % self.num_docs
            fout.write(utils.to_utf8(numdocs_line))
            if sort_by_word:
                for token, tokenid in sorted(self.token2id.items()):
                    line = "%i\t%s\t%i\n" % (tokenid, token, self.dfs.get(tokenid, 0))
                    fout.write(utils.to_utf8(line))
            else:
                for tokenid, freq in sorted(self.dfs.items(), key=lambda item: -item[1]):
                    line = "%i\t%s\t%i\n" % (tokenid, self[tokenid], freq)
                    fout.write(utils.to_utf8(line))

    def merge_with(self, other):
        old2new = {}
        for other_id, other_token in other.items():
            if other_token in self.token2id:
                new_id = self.token2id[other_token]
            else:
                new_id = len(self.token2id)
                self.token2id[other_token] = new_id
                self.dfs[new_id] = 0
            old2new[other_id] = new_id
            try:
                self.dfs[new_id] += other.dfs[other_id]
            except Exception:
                pass
        try:
            self.num_docs += other.num_docs
            self.num_nnz += other.num_nnz
            self.num_pos += other.num_pos
        except Exception:
            pass
        return gensim.models.VocabTransform(old2new)

    def patch_with_special_tokens(self, special_token_dict):
        possible_ids = []
        for token, idx in special_token_dict.items():
            if token in self.token2id and self.token2id[token] == idx:
                continue
            if token in self.token2id and self.token2id[token] != idx:
                possible_ids.append(self.token2id[token])
                del self.token2id[token]
            old_token = self[idx]
            self.token2id[token] = idx
            self.token2id[old_token] = possible_ids.pop() if \
                                       len(possible_ids) > 0 else len(self.token2id) - 1
        self.id2token = {}

    @staticmethod
    def load_from_text(fname):
        result = Dictionary()
        with utils.open(fname, 'rb') as f:
            for lineno, line in enumerate(f):
                line = utils.to_unicode(line)
                if lineno == 0:
                    if line.strip().isdigit():
                        result.num_docs = int(line.strip())
                        continue
                    else:
                        logging.warning("Text does not contain num_docs on the first line.")
                try:
                    wordid, word, docfreq = line[:-1].split('\t')
                except Exception:
                    raise ValueError("invalid line in dictionary file %s: %s"
                                     % (fname, line.strip()))
                wordid = int(wordid)
                if word in result.token2id:
                    raise KeyError('token %s is defined as ID %d and as ID %d' % (word, wordid, result.token2id[word]))
                result.token2id[word] = wordid
                result.dfs[wordid] = int(docfreq)
        return result

    def most_common(self, n: Optional[int] = None) -> List[Tuple[str, int]]:
        most_common = [
            (self[word], count)
            for word, count
            in sorted(self.cfs.items(), key=lambda x: (-x[1], x[0]))[:n]
        ]
        return most_common

    @staticmethod
    def from_corpus(corpus, id2word=None):
        result = Dictionary()
        max_id = -1
        for docno, document in enumerate(corpus):
            if docno % 10000 == 0:
                logger.info("adding document #%i to %s", docno, result)
            result.num_docs += 1
            result.num_nnz += len(document)
            for wordid, word_freq in document:
                max_id = max(wordid, max_id)
                result.num_pos += word_freq
                result.dfs[wordid] = result.dfs.get(wordid, 0) + 1

        if id2word is None:
            result.token2id = {str(i): i for i in range(max_id + 1)}
        else:
            result.token2id = {utils.to_unicode(token): idx for idx, token in id2word.items()}
        for idx in result.token2id.values():
            result.dfs[idx] = result.dfs.get(idx, 0)

        logger.info(
            "built %s from %i documents (total %i corpus positions)",
            result, result.num_docs, result.num_pos
        )
        return result
